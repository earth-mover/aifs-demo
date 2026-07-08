"""
Run AIFS forecasts using Brightband's curated ECMWF IFS initial conditions
from the Earthmover data marketplace.

Inference code adapted from https://huggingface.co/ecmwf/aifs-single-1.0/blob/main/run_AIFS_v1.ipynb
Data processing adapted from Brightband's reference notebook:
https://colab.research.google.com/drive/1rmKPe2oeF05sJ__sCj3qEOo4fjRho9Vl
"""

import datetime
import queue
import threading
from collections import defaultdict

import click
import numpy as np
import pandas as pd
import xarray as xr
from arraylake import Client
import earthkit.regrid as ekr
from tqdm import tqdm

# A subscription to Brightband's "ECMWF IFS Initial Conditions (open)"
# marketplace listing: https://app.earthmover.io/marketplace/697162921880507a6587c31b
DEFAULT_IC_REPO = "vandelay-industries/my-ifs-ics"
DEFAULT_TARGET_REPO = "vandelay-industries/aifs-outputs"

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Brightband variable names that differ from AIFS parameter names
RENAME_MAP = {"u10": "10u", "v10": "10v", "d2m": "2d", "t2m": "2t", "z_sfc": "z"}

# Single-level variables (surface + soil)
SFC_VARS = [
    "u10",
    "v10",
    "d2m",
    "t2m",
    "msl",
    "skt",
    "sp",
    "tcw",
    "stl1",
    "stl2",
    "swvl1",
    "swvl2",
]
# Static/constant fields; not yet merged to the main branch of the
# Brightband repo (see STATIC_BRANCH)
STATIC_VARS = ["lsm", "z_sfc", "slor", "sdor"]
STATIC_BRANCH = "add-static-vars"

# Pressure-level variables; same names in Brightband and AIFS.
# z is already geopotential (m^2/s^2), no gh -> z conversion needed.
PL_VARS = ["z", "t", "u", "v", "w", "q"]


def open_initial_conditions(
    ic_repo_name: str = DEFAULT_IC_REPO,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Open the Brightband IFS initial conditions as a pair of
    (time-varying, static) xarray datasets."""
    client = Client()
    repo = client.get_repo(ic_repo_name)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False)

    if all(var in ds for var in STATIC_VARS):
        ds_static = ds[STATIC_VARS]
    else:
        static_session = repo.readonly_session(branch=STATIC_BRANCH)
        ds_static = xr.open_zarr(
            static_session.store, zarr_format=3, consolidated=False
        )[STATIC_VARS]

    # the earthkit regrid weights expect longitude in 0:360;
    # the repo stores -180:180
    def shift_lon(ds):
        return ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")

    return shift_lon(ds), shift_lon(ds_static)


def get_gpu_regridder(source_grid, target_grid, method="linear", device="cuda"):
    """Create a GPU regridder using weights from the Earthkit regrid module."""

    # Note: import torch here to avoid having to have pytorch
    # installed in the environment by default.
    import torch

    class GPU_Regridder:
        def __init__(self, source_grid, target_grid, method="linear"):
            weights_csr, self.target_shape = ekr.db.find(
                source_grid, target_grid, method
            )
            self.device = device
            self.weights = torch.sparse_csr_tensor(
                torch.from_numpy(weights_csr.indptr),
                torch.from_numpy(weights_csr.indices),
                torch.from_numpy(weights_csr.data),
                size=weights_csr.shape,
            ).to(device)

        def regrid(self, data):
            tensor = torch.from_numpy(data.astype("f8").reshape(-1)).to(self.device)
            regridded = self.weights.matmul(tensor)
            return regridded.cpu().numpy().astype("f4").reshape(self.target_shape)

    return GPU_Regridder(source_grid, target_grid, method)


def fetch_initial_conditions(
    date: datetime.datetime,
    ds: xr.Dataset,
    ds_static: xr.Dataset,
    regridder,
) -> dict[str, np.ndarray]:
    """Extract and regrid all AIFS input fields from the Brightband dataset.

    Selects the analysis (lead_time=0) for the given date and the previous
    6-hourly step, and regrids from the native 0.25 degree lat/lon grid to the
    model's N320 Gaussian grid.

    Returns a dict mapping AIFS parameter names to arrays of shape
    (2, N320_points).
    """
    # static fields are the same for both time steps, regrid them once
    static_fields = {
        RENAME_MAP.get(var, var): regridder.regrid(ds_static[var].values)
        for var in STATIC_VARS
    }

    # AIFS requires two consecutive 6-hourly time steps as initial conditions
    all_fields = defaultdict(list)
    for init_time in [date - datetime.timedelta(hours=6), date]:
        ds_t = ds.sel(
            init_time=np.datetime64(init_time.replace(tzinfo=None)),
            lead_time=np.timedelta64(0, "ns"),
        )
        for var in SFC_VARS:
            all_fields[RENAME_MAP.get(var, var)].append(
                regridder.regrid(ds_t[var].values)
            )
        for var in PL_VARS:
            # load all levels of this variable in one read
            data = ds_t[var].sel(level=LEVELS).values
            for n, level in enumerate(LEVELS):
                all_fields[f"{var}_{level}"].append(regridder.regrid(data[n]))
        for name, values in static_fields.items():
            all_fields[name].append(values)

    return {name: np.stack(values) for name, values in all_fields.items()}


def datetime_to_str(date: datetime.datetime) -> str:
    """Helper function to convert a datetime to a string."""
    assert date.tzinfo == datetime.UTC
    assert date.minute == date.second == date.microsecond == 0
    assert date.hour in [0, 6, 12, 18]
    return date.strftime("%Y-%m-%d/%Hz")


def state_to_xarray(state, regridder, include_pressure_levels=False):
    fields = state["fields"]
    dims = ("valid_time", "lat", "lon")
    lat = 90 - 0.25 * np.arange(721)
    lon = 0.25 * np.arange(1440)
    pressure = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    ds = xr.Dataset(
        {
            vname: (
                dims,
                regridder.regrid(array)[None, :, :],
                # dsa.from_delayed(regrid_delayed(array), shape=(1, 721, 1440), dtype="f4")
            )
            for vname, array in fields.items()
        },
        coords={
            "valid_time": (
                "valid_time",
                [state["date"]],
                {"axis": "T", "standard_name": "time"},
            ),
            "lat": ("lat", lat, {"standard_name": "latitude", "axis": "Y"}),
            "lon": ("lon", lon, {"standard_name": "longitude", "axis": "X"}),
            "pressure": pressure,
        },
    )
    ds.valid_time.encoding.update(
        {"units": "hours since 1970-01-01T00:00:00", "chunks": (1200,)}
    )

    to_drop = []
    for pvar in ["q", "t", "u", "v", "w", "z"]:
        vnames = [f"{pvar}_{plev}" for plev in pressure]
        if include_pressure_levels:
            ds[pvar] = xr.concat(
                [ds[vname] for vname in vnames], dim="pressure"
            ).transpose("valid_time", ...)
        to_drop.extend(vnames)

    ds = ds.drop_vars(to_drop)

    return ds


def run_single_forecast(
    date: datetime.datetime,
    ic_repo_name: str,
    target_repo_name: str,
) -> None:
    """
    Run the forecast for a given date with its own independent session.
    """
    from anemoi.inference.runners.simple import SimpleRunner
    from anemoi.inference.outputs.printer import print_state
    import torch

    client = Client()
    ds, ds_static = open_initial_conditions(ic_repo_name)
    target_repo = client.get_or_create_repo(target_repo_name)
    target_session = target_repo.writable_session("main")

    checkpoint = {"huggingface": "ecmwf/aifs-single-1.0"}
    runner = SimpleRunner(checkpoint, device="cuda")

    print("setting up regridders")
    input_regridder = get_gpu_regridder({"grid": (0.25, 0.25)}, {"grid": "N320"})
    output_regridder = get_gpu_regridder({"grid": "N320"}, {"grid": (0.25, 0.25)})

    print("loading initial conditions for", date)
    fields = fetch_initial_conditions(date, ds, ds_static, input_regridder)

    date_no_tz = date.replace(tzinfo=None)
    input_state = dict(date=date_no_tz, fields=fields)

    # we put data that we want to write into a queue
    q = queue.Queue()
    lock = threading.Lock()

    def worker():
        while True:
            (ds, store, group_name, kwargs) = q.get()
            # lock is probably unncessary
            with lock:
                ds.to_zarr(
                    store, group=group_name, zarr_format=3, consolidated=False, **kwargs
                )
            q.task_done()

    # a separate thread for I/O to avoid blocking the main loop
    threading.Thread(target=worker, daemon=True).start()

    print("starting forecast loop")
    kwargs = {"mode": "w"}
    # main forecast loop
    for n, state in enumerate(runner.run(input_state=input_state, lead_time=48)):
        print_state(state)
        ds_out = state_to_xarray(state, regridder=output_regridder).chunk()
        group = datetime_to_str(date)
        if n > 0:
            kwargs = {"mode": "a", "append_dim": "valid_time"}
        q.put((ds_out, target_session.store, group, kwargs))

    q.join()  # wait for all I/O tasks to finish

    # clear GPU memory
    torch.cuda.empty_cache()

    # Commit this forecast's data
    target_session.commit(f"forecast for {date.strftime('%Y-%m-%d %H:%M')}")


@click.group()
def cli():
    """AIFS forecast CLI application."""
    pass


@cli.command()
@click.argument("start_date")
@click.argument("end_date")
@click.option("--ic-repo-name", default=DEFAULT_IC_REPO)
@click.option("--target-repo-name", default=DEFAULT_TARGET_REPO)
def forecast(start_date: str, end_date: str, ic_repo_name: str, target_repo_name: str):
    dates = [
        item.to_pydatetime()
        for item in pd.date_range(start_date, end_date, freq="6h", tz=datetime.UTC)
    ]

    # Run forecasts sequentially, each with its own session
    for date in tqdm(dates, desc="running forecasts"):
        run_single_forecast(date, ic_repo_name, target_repo_name)

    print("All forecasts completed - each was committed individually")


if __name__ == "__main__":
    cli()
