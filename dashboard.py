import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    _TODO: add Earthmover logo here_

    # AIFS Forecast Demo Dashboard

    ## Introduction

    This is a demo dashboard for the AIFS Forecasting demo. At this point, we've already generated an Icechunk repository with a selection of forecasts. The dashboard will allow you to select a forecast to view.

    ## How to use this dashboard

    Feel free to use the dashboard in Marimo's "App View" mode. Or switch over to "Edit Mode" and work directly with [Icechunk](https://icechunk.io/) and [Xarray](https://xarray.dev/).


    ## Getting started

    Before we can begin, we'll authenticate with Earthmover. This will give us access to the AIFS forecast repository. We'll run:

    ```python
    from arraylake import Client

    client = Client()
    client.login()
    ```

    This will prompt you for you to login with Earthmover. Follow the instructions on the screen.
    """
    )
    return


@app.cell
def _():
    import functools

    import marimo as mo
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np

    from arraylake import Client
    import xarray as xr
    import zarr
    import icechunk as ic


    client = Client()

    with mo.capture_stdout() as login_buffer:
        client.login()
    mo.md(f"```\n{login_buffer.getvalue()}\n```")
    return ccrs, cfeature, client, functools, mo, plt, xr, zarr


@app.cell
def _(mo):
    mo.md(
        r"""
    We've written the forecast data to the `aifs-outputs` repository. We'll now use the Arraylake client to access the Icechunk repository. 

    ```python
    repo = client.get_repo('earthmover-public/aifs-outputs')
    ```
    """
    )
    return


@app.cell
def _(client, zarr):
    repo = client.get_repo('earthmover-public/aifs-outputs')
    session = repo.readonly_session('main')
    root = zarr.open_group(session.store, mode='r', zarr_format=3)

    # 👇 don't worry about this warning
    return root, session


@app.cell
def _(root):
    dates = list(root.group_keys())
    dates.sort(reverse=True)
    forecasts = ['00z', '06z', '12z', '18z']
    return dates, forecasts


@app.cell
def _(functools, session, xr, zarr):
    @functools.lru_cache(maxsize=10)
    def get_forecast_ds(group: str | None) -> xr.Dataset | None:
        try:
            fds = xr.open_dataset(session.store, group=group, engine='zarr', zarr_format=3, chunks=None)
        except zarr.errors.GroupNotFoundError:
            print(f"❌ Forecast group not found: {group}")
            return None
        else:
            print(f"✅ opened forecast dataset: {group}")
            return fds
    return (get_forecast_ds,)


@app.cell
def _(ccrs, cfeature, plt, xr):
    def make_cartopy_plot(
        arr: xr.DataArray,
        projection=ccrs.PlateCarree(),
        cmap='viridis',
        title=None,
        **kwargs
    ):
        """
        Creates a plot of an xarray.DataArray with cartopy coastlines and boundaries.

        Args:
            arr (xr.DataArray): The data to plot. Must have 1D 'lat' and 'lon' dims.
            projection (ccrs.Projection, optional): The cartopy projection for the map.
                                                    Defaults to ccrs.PlateCarree().
            cmap (str, optional): The colormap to use. Defaults to 'viridis'.
            title (str, optional): The plot title. Defaults to arr.name.
            **kwargs: Additional keyword arguments passed to arr.plot().
        """
        # Create the figure and axes with the specified projection
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection=projection)

        # The data coordinates are in lat/lon, so we use PlateCarree transform
        # This tells cartopy how to interpret the data's coordinates
        data_transform = ccrs.PlateCarree()

        # Plot the data. xarray's plot function is cartopy-aware.
        arr.plot(
            ax=ax,
            transform=data_transform,
            cmap=cmap,
            add_colorbar=True,
            cbar_kwargs={'shrink': 0.7, 'orientation': 'horizontal', 'pad': 0.05},
            **kwargs
        )

        # Add geographic features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")

        # Add gridlines with labels for better context
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title('')

        return fig
    return (make_cartopy_plot,)


@app.cell
def _(dates, forecasts, mo):
    date_selector = mo.ui.dropdown(
        options=dates,
        value=dates[0] if dates else None,
        label="Select Forecast Date"
    )

    time_selector = mo.ui.dropdown(
        options=forecasts,
        value=forecasts[0],
        label="Select Forecast Time"
    )
    return date_selector, time_selector


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Explore the forecasts

    We can easily open any of the existing forecasts with Xarray and plot them. We'll first make a map of a choosen forecast, then we'll compare the time series for multiple forecasts.

    ### Make a forecast map

    This section of the dashboard demonstrates how to plot a map from any forecast in the archive. Use the drop down options to select a forecast date, time, variable, forecast valid-time and region to plot.

    Note the structure of the Xarray dataset. Each of the 24 data variables is a 3 dimensional data cube, with `(valid_time, lat, lon)` as dimensions.
    """
    )
    return


@app.cell
def _(date_selector, mo, time_selector):
    selected_forecast = f"{date_selector.value}/{time_selector.value}" if date_selector.value else None

    mo.vstack([
        mo.hstack([date_selector, time_selector]),
        mo.md(f"**Selected forecast:** `{selected_forecast}`") if selected_forecast else mo.md("No forecast selected")
    ])
    return (selected_forecast,)


@app.cell
def _(get_forecast_ds, selected_forecast):
    ds_select = get_forecast_ds(selected_forecast)
    ds_select
    return (ds_select,)


@app.cell
def _(ds_select, mo):
    variables = set(ds_select.variables) - {'lon', 'lat', 'time', 'valid_time', 'pressure'}
    var_selector = mo.ui.dropdown(
        options=list(variables),
        value='2t',
        label="Select Variable to plot"
    )

    valid_times = ds_select.valid_time.dt.strftime("%Y-%m-%dT%H").values.tolist()
    valid_time_selector = mo.ui.dropdown(
        options=valid_times,
        value=valid_times[0],
        label="Select Valid Time"
    )

    regions = {
        "Global": {
            "lat": slice(-90, 90),
            "lon": slice(0, 360),
        },
        "US": {
            "lat": slice(49, 24.5),
            "lon": slice(360 - 125, 360 - 66.9),
        },
        "Europe": {
            "lat": slice(72, 34),           # Norway down to Mediterranean
            "lon": slice(0, 40),     # Portugal to western Russia
        },
        "China": {
            "lat": slice(54, 18),           # Heilongjiang to Hainan
            "lon": slice(73, 135),          # Xinjiang to east coast
        },
        "Australia": {
            "lat": slice(-10, -44),         # Top End down to Tasmania
            "lon": slice(112, 154),         # Western Australia to east coast
        },
        "SouthAmerica": {
            "lat": slice(12, -56),          # Colombia down to Tierra del Fuego
            "lon": slice(285, 330),         # Chile/Argentina to Brazil
        },
        "Africa": {
            "lat": slice(38, -35),          # Mediterranean to Cape of Good Hope
            "lon": slice(0, 52),    # Morocco to Horn of Africa
        },
        "MiddleEast": {
            "lat": slice(40, 12),           # Turkey down to Yemen
            "lon": slice(32, 60),           # Egypt to Iran
        },
    }


    region_selector = mo.ui.dropdown(
        options=list(regions),
        value='US',
        label="Select Region"
    )
    return (
        region_selector,
        regions,
        valid_time_selector,
        var_selector,
        variables,
    )


@app.cell
def _(ds_select, var_selector):
    var = ds_select[var_selector.value]
    return (var,)


@app.cell
def _(mo, region_selector, valid_time_selector, var_selector):
    mo.vstack([
        mo.hstack([var_selector, valid_time_selector, region_selector]),
        mo.md(f"**Plotting:** variable `{var_selector.value}` for `{valid_time_selector.value}`"),
    ])
    return


@app.cell
def _(make_cartopy_plot, region_selector, regions, valid_time_selector, var):
    fig = make_cartopy_plot(var.sel(valid_time=valid_time_selector.value, **regions[region_selector.value]))
    fig
    return


@app.cell
def _(dates, forecasts, mo, variables):
    # Two input boxes for latitude and longitude
    lat_input = mo.ui.text(value="237.58", label="Latitude")
    lon_input = mo.ui.text(value="37.77", label="Longitude")

    # And a dropdown for the variable to plot
    var_selector2 = mo.ui.dropdown(
        options=list(variables),
        value='2t',
        label="Select Variable to plot"
    )


    # And a multiselect for the forecast(s)
    all_forecasts = []
    for date in dates:
        all_forecasts.extend([f'{date}/{hour}' for hour in forecasts])
    forecast_multi_selector = mo.ui.multiselect(
        options=all_forecasts,
        label="Select forecast(s)",
        value=all_forecasts[:1],
        max_selections=10,
    )
    return forecast_multi_selector, lat_input, lon_input, var_selector2


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Compare forecasts

    In our setup, we can run AIFS four times a day. In this section, we'll explore how the successive forecasts compare at a single point in space. Choose a latitude, longitude, variable, and one or more forecasts to compare.
    """
    )
    return


@app.cell
def _(forecast_multi_selector, lat_input, lon_input, mo, var_selector2):
    mo.hstack([lat_input, lon_input, forecast_multi_selector, var_selector2])
    return


@app.cell
def _(
    forecast_multi_selector,
    get_forecast_ds,
    lat_input,
    lon_input,
    plt,
    var_selector2,
):
    # Plot the forecast at a single point
    fig2 = plt.figure(figsize=(10, 4))
    ax2 = fig2.add_subplot(111)

    for f in forecast_multi_selector.value:
        ds = get_forecast_ds(f)
        if ds is not None:
            da = ds[var_selector2.value].sel(lon=float(lon_input.value), lat=float(lat_input.value), method='nearest')
            da.plot(ax=ax2, label=f)

    ax2.legend()
    fig2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Explore on your own

    From here, you can explore the repository on your own. We recomment switching into edit mode (`command + .`) and writing some Xarray code yourself. Below are some tips to get you started.

    ### Opening a Forecast with Xarray

    Each AIFS is stored as a separate group in our Icechunk repository. We can open these in Xarray as follows:

    ```python
    my_forecast_group = "2025-04-01/00z"
    my_ds = xr.open_zarr(session.store, group=my_forecast_group)
    ```

    ### Plotting a single variable

    ```
    my_forecast_group = "2025-04-01/00z"
    my_ds = xr.open_zarr(session.store, group=my_forecast_group)
    my_ds
    ```

    ### Suggested activities

    Once you have the basics down, try a few of these:

    1. Make a map of the total cloud cover (`tcc`) for this afternoon in the US.
    2. Make a map of the change in forecast temperature between two forecasts.
    3. Extract the time series of the forecast for the city/town where you grew up.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
