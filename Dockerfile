# GPU software environment for the AIFS forecast workflow, for use with
# Coiled (https://docs.coiled.io/user_guide/software/docker.html).
#
# Built entirely from the rendered conda-lock explicit lockfile - no solver
# runs at build time and the resulting image is fully deterministic. The lock
# includes dask and distributed (required by Coiled) plus the exact pinned
# pip wheels (as "# pip" lines, natively supported by micromamba install).
#
# Regenerate env/conda-linux-64.lock after changing env/environment.yaml:
#   uvx conda-lock lock -f env/environment.yaml -p linux-64 \
#       --virtual-package-spec env/virtual-packages.yaml \
#       --lockfile env/conda-lock.yml
#   uvx conda-lock render -p linux-64 --kind explicit env/conda-lock.yml
#   mv conda-linux-64.lock env/conda-linux-64.lock
#
# Build (from repo root):
#   docker build --platform linux/amd64 -t aifs-demo .

FROM mambaorg/micromamba:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER env/conda-linux-64.lock /tmp/conda-linux-64.lock

RUN micromamba install --name base --yes --file /tmp/conda-linux-64.lock \
    && micromamba clean --all --yes

# micromamba install silently ignores the "# pip" lines in explicit lockfiles,
# so install the pip layer separately from the pins extracted from the same
# conda-lock solve. --no-deps: the pin list is the complete closure.
COPY --chown=$MAMBA_USER:$MAMBA_USER env/requirements-locked.txt /tmp/requirements-locked.txt

RUN micromamba run -n base python -m pip install --no-deps --no-cache-dir \
    -r /tmp/requirements-locked.txt

# The micromamba entrypoint activates the base environment, so `docker run
# <image> python` works, which is what Coiled expects (use_entrypoint=True).
