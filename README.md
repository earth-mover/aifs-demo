# AIFS Forecasting Demo

> [!NOTE]
> This repo is for demonstration purposes only. It does not aspire to be a maintained package. If you want to build on top of it, fork this repo and modify it to your needs.

License: Apache 2.0

Some of the code has been adapted from an [ECWMF Notebook](https://huggingface.co/ecmwf/aifs-single-1.0) under the Apache 2.0 license.
The data processing follows [Brightband's reference notebook](https://colab.research.google.com/drive/1rmKPe2oeF05sJ__sCj3qEOo4fjRho9Vl).

## Data Source

Initial conditions come from Brightband's
[ECMWF IFS Initial Conditions (open)](https://app.earthmover.io/marketplace/697162921880507a6587c31b)
listing on the Earthmover data marketplace. To use it, subscribe to the listing
from your Arraylake org; this repo assumes a subscription repo named
`vandelay-industries/my-ifs-ics`.

The dataset is a rolling cube of ECMWF IFS HRES analysis states (0.25°,
6-hourly, 13 pressure levels) containing everything needed to initialize
MLWP models like AIFS. Static fields (`lsm`, `z_sfc`, `slor`, `sdor`)
currently live on the `add-static-vars` branch of the Brightband repo and are
picked up from there automatically until they are merged to `main`.

## Usage

This code is packaged as a command-line script. Run it from your laptop; the
GPU work is dispatched automatically to a Coiled GPU VM (or run it directly
on a GPU machine with `--local`). At each forecast initialization time, it:

1. Reads two consecutive 6-hourly analysis states from the Brightband dataset
2. Regrids them from 0.25° to the model's N320 Gaussian grid on the GPU
3. Runs the [aifs-single-1.0](https://huggingface.co/ecmwf/aifs-single-1.0) model with `anemoi-inference`
4. Regrids the outputs back to 0.25° on the GPU and writes them to an Arraylake repo

```bash
% python main.py forecast --help
Usage: main.py forecast [OPTIONS] START_DATE END_DATE

Options:
  --ic-repo-name TEXT
  --target-repo-name TEXT
  --local                  Run on this machine (requires a CUDA GPU) instead
                           of a Coiled GPU VM.
  --help                   Show this message and exit.
```

For example:

```bash
python main.py forecast 2026-07-07T12:00 2026-07-07T12:00
```

Authentication uses your Arraylake login, or set the `ARRAYLAKE_TOKEN`
environment variable to a service-account token (useful for headless runs);
the driver forwards the token to the Coiled VM automatically.

### The Coiled software environment

The GPU software environment is a Docker image (see `Dockerfile`) built from
the fully-pinned `env/conda-lock.yml` and pushed to ECR, then registered with
Coiled as the `aifs-docker` environment. A container image is used because
Coiled's remote builder currently cannot build conda environments containing
CUDA packages (pytorch, flash-attn). See `create_software_environments.py`
for the full rebuild recipe, and `env/environment.yaml` for the top-level
package specification.

Register the environment (once, after pushing the image):

```bash
python create_software_environments.py
```

For interactive work, start a Jupyter session on a GPU VM (see
`run_notebook.sh`) and use the `run-aifs-earthmover.ipynb` notebook.

## Dashboard demo

`dashboard.py` includes a Marimo notebook that can be used to analyze the outputs of the AIFS forecast. 

Run it locally:

```
marimo edit dashboard.py
```

Run it from GitHub directly:

```
uvx marimo edit --sandbox https://github.com/earth-mover/aifs-demo/blob/main/dashboard.py
```
