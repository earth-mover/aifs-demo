"""
Run ETL jobs to ingest AIFS data into Arraylake.
Ingestion code adapted from https://huggingface.co/ecmwf/aifs-single-1.0/blob/main/run_AIFS_v1.ipynb
"""

import datetime
import queue
import threading

import pandas as pd
import click
import icechunk.distributed
import xarray as xr
import zarr
import icechunk
from arraylake import Client
import coiled
from coiled import Cluster
import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
from dask.distributed import as_completed
from tqdm import tqdm

from earthkit.data import config

config.set("cache-policy", "off")


PARAM_SFC = [
    "10u",
    "10v",
    "2d",
    "2t",
    "msl",
    "skt",
    "sp",
    "tcw",
    "lsm",
    "z",
    "slor",
    "sdor",
]
PARAM_SOIL = ["vsw", "sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]


def get_data_for_date(
    date: datetime.datetime, param: str, levelist: list[int] = []
) -> dict[str, np.ndarray]:
    fields = {}
    data = ekd.from_source(
        "ecmwf-open-data", date=date, param=param, levelist=levelist, source="aws"
    )
    for f in data:
        # Open data is between -180 and 180, we need to shift it to 0-360
        array = f.to_numpy(dtype="f4")  # no need for 64 bit precision
        assert array.shape == (721, 1440)
        values = np.roll(array, -array.shape[1] // 2, axis=1)
        # Interpolate the data to from 0.25 to N320
        values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
        # no need for 64-bit precision
        values = values.astype("f4")
        name = (
            f"{f.metadata('param')}_{f.metadata('levelist')}"
            if levelist
            else f.metadata("param")
        )
        fields[name] = values
    return fields


def get_all_data(date: datetime.datetime) -> dict[str, np.ndarray]:
    data_dict = {}
    for param in PARAM_SFC:
        data_dict.update(get_data_for_date(date, param))
    for param in PARAM_SOIL:
        data_dict.update(get_data_for_date(date, param, SOIL_LEVELS))
    for param in PARAM_PL:
        data_dict.update(get_data_for_date(date, param, LEVELS))
    return data_dict


def stack_fields(data_dict: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Turn many numpy arrays into a single numpy array."""
    # merge the dicts into a single dict
    names = list(data_dict.keys())
    arrays = [data_dict[name] for name in names]
    shape = arrays[0].shape
    dtype = arrays[0].dtype
    assert len(shape) == 1
    assert all(v.shape == shape for v in arrays)
    # stack the arrays into a single array with a new dimension
    stacked = np.stack(arrays, axis=0)
    return names, stacked


def store_data(group: zarr.Group, variable_names: list[str], data: np.ndarray):
    assert data.ndim == 2
    nvars = len(variable_names)
    assert data.shape[0] == nvars
    npoints = data.shape[1]
    var_array = group.create_array(
        "variable",
        dtype=str,
        shape=(nvars,),
        chunks=(nvars,),
        compressors=[],
        dimension_names=["variable"],
    )
    var_array[:] = variable_names
    data_array = group.create_array(
        "fields",
        dtype=data.dtype,
        shape=data.shape,
        chunks=(10, npoints),
        dimension_names=["variable", "point"],
    )
    data_array[:] = data


def datetime_to_str(date: datetime.datetime) -> str:
    """Helper function to convert a datetime to a string."""
    assert date.tzinfo == datetime.UTC
    assert date.minute == date.second == date.microsecond == 0
    assert date.hour in [0, 6, 12, 18]
    return date.strftime("%Y-%m-%d/%Hz")


def get_and_store_date(
    date: datetime.datetime, session: icechunk.Session
) -> icechunk.Session:

    store = session.store
    group_name = datetime_to_str(date)
    group = zarr.group(store=store, path=group_name, overwrite=True)

    data_dict = get_all_data(date)
    names, stacked = stack_fields(data_dict)

    store_data(group, names, stacked)
    return session


def get_gpu_regridder(source_grid, target_grid, method="linear"):
    """Create a GPU regridder using weights the Earthkit regrid module.
    Note: we define this function inline to avoid having to have pytorch
    installed in the environment by default.
    """

    import torch

    class GPU_Regridder:

        def __init__(self, source_grid, target_grid, method="linear"):
            weights_csr, self.target_shape = ekr.db.find(
                source_grid, target_grid, method
            )
            self.weights = torch.sparse_csr_tensor(
                torch.from_numpy(weights_csr.indptr),
                torch.from_numpy(weights_csr.indices),
                torch.from_numpy(weights_csr.data),
                size=weights_csr.shape,
            ).cuda()

        def regrid(self, data):
            tensor = torch.from_numpy(data.astype("f8")).cuda()
            regridded = self.weights.matmul(tensor)
            return regridded.cpu().numpy().astype("f4").reshape(self.target_shape)

    return GPU_Regridder(source_grid, target_grid, method)


def fetch_initial_conditions(
    date: datetime.datetime, session: icechunk.Session
) -> dict[str, np.ndarray]:
    group_prev = zarr.open_group(
        session.store,
        zarr_format=3,
        path=datetime_to_str(date - datetime.timedelta(hours=6)),
        mode="r",
    )
    group_curr = zarr.open_group(
        session.store, zarr_format=3, path=datetime_to_str(date), mode="r"
    )

    vnames_curr = group_curr["variable"][:]
    vnames_prev = group_prev["variable"][:]
    np.testing.assert_equal(vnames_curr, vnames_prev)

    fields_prev = group_prev["fields"][:]
    fields_curr = group_curr["fields"][:]
    data = np.stack([fields_prev, fields_curr], axis=1)

    # tweak data to conform with AIFS input format
    mapping = {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}

    def maybe_rename_vname(vname):
        if vname in mapping:
            return mapping[vname]
        return vname

    fields = {
        maybe_rename_vname(vnames_curr[n]): data[n] for n in range(len(vnames_curr))
    }

    # convert to geopotential height
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665

    return fields


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
            "valid_time": ("valid_time", [state["date"]], {"axis": "T"}),
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
    source_session: icechunk.Session,
    target_session: icechunk.Session,
) -> icechunk.Session:
    """
    Run the forecast for a given date.
    """
    from anemoi.inference.runners.simple import SimpleRunner
    from anemoi.inference.outputs.printer import print_state

    checkpoint = {"huggingface": "ecmwf/aifs-single-1.0"}
    runner = SimpleRunner(checkpoint, device="cuda")

    print("loading initial conditions for", date)
    fields = fetch_initial_conditions(date, source_session)

    date_no_tz = date.replace(tzinfo=None)
    input_state = dict(date=date_no_tz, fields=fields)

    print("setting up regridder")
    regridder = get_gpu_regridder({"grid": "N320"}, {"grid": (0.25, 0.25)})

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
        ds = state_to_xarray(state, regridder=regridder).chunk()
        group = datetime_to_str(date)
        if n > 0:
            kwargs = {"mode": "a", "append_dim": "valid_time"}
        q.put((ds, target_session.store, group, kwargs))

    q.join()  # wait for all I/O tasks to finish

    return target_session


@click.group()
def cli():
    """AIFS ETL CLI application."""
    pass


@cli.command()
@click.argument("start_date")
@click.argument("end_date")
@click.option("--repo-name", default="earthmover-public/aifs-initial-conditions")
def ingest(start_date: str, end_date: str, repo_name: str):
    dates = [
        item.to_pydatetime()
        for item in pd.date_range(start_date, end_date, freq="6h", tz=datetime.UTC)
    ]

    client = Client()
    repo = client.get_or_create_repo(repo_name)
    session = repo.writable_session("main")

    cluster = Cluster(
        name="aifs-etl",
        software="aifs-etl",
        n_workers=[1, 500],
        region="us-east-1",
        shutdown_on_close=False,
        arm=False,
        spot_policy="spot_with_fallback",
        idle_timeout="10m",
        worker_vm_types=["m4.large"],
    )
    dclient = cluster.get_client()

    # autoscaling doesn't seem to work well, use our own heuristic
    cluster.scale(len(dates) // 4)

    with session.allow_pickling():
        futures = [
            dclient.submit(get_and_store_date, date, session=session)
            for date in tqdm(dates, desc="scheduling tasks")
        ]

    results = [
        result
        for fut, result in tqdm(
            as_completed(futures, with_results=True),
            desc="running tasks",
            total=len(futures),
        )
    ]
    merged_session = icechunk.distributed.merge_sessions(list(results))
    merged_session.commit(f"wrote {start_date} to {end_date}")


@cli.command()
@click.argument("start_date")
@click.argument("end_date")
@click.option("--ic-repo-name", default="earthmover-public/aifs-initial-conditions")
@click.option("--target-repo-name", default="earthmover-public/aifs-outputs")
def forecast(start_date: str, end_date: str, ic_repo_name: str, target_repo_name: str):

    dates = [
        item.to_pydatetime()
        for item in pd.date_range(start_date, end_date, freq="6h", tz=datetime.UTC)
    ]

    client = Client()
    ic_repo = client.get_or_create_repo(ic_repo_name)
    ic_session = ic_repo.readonly_session("main")
    target_repo = client.get_or_create_repo(target_repo_name)
    target_session = target_repo.writable_session("main")

    # scale workers between 1 and 10
    n_workers = min(max(len(dates) // 5, 1), 10)

    cluster = Cluster(
        name="aifs-forecast",
        software="aifs-conda",
        n_workers=(1, 10),
        region="us-east-1",
        shutdown_on_close=False,
        arm=False,
        spot_policy="spot_with_fallback",
        idle_timeout="10m",
        worker_vm_types=["g6e.2xlarge", "g6e.xlarge"],
        worker_options={"nthreads": 1},  # one thread per worker to avoid GPU contention
    )
    cluster.scale(n_workers)

    dclient = cluster.get_client()

    with ic_session.allow_pickling():
        with target_session.allow_pickling():
            futures = [
                dclient.submit(
                    run_single_forecast,
                    date,
                    source_session=ic_session,
                    target_session=target_session,
                )
                for date in tqdm(dates, desc="scheduling forecast tasks")
            ]

    results = [
        result
        for fut, result in tqdm(
            as_completed(futures, with_results=True),
            desc="running tasks",
            total=len(futures),
        )
    ]
    print("merging sessions")
    merged_session = icechunk.distributed.merge_sessions(list(results))
    print("committing results")
    merged_session.commit(f"wrote forecast for {start_date} to {end_date}")


if __name__ == "__main__":
    cli()
