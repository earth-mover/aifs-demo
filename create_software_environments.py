"""Register the Coiled software environment for the AIFS forecast workflow.

The environment is a Docker image built from a conda-lock file (see
Dockerfile). We use a container image rather than a Coiled-built environment
because Coiled's remote builder currently cannot produce this environment:
its rattler-based solver does not provide the __cuda virtual package (so
conda CUDA packages like pytorch and flash-attn are unsolvable there), it
silently drops direct-URL pip requirements, and its conda-lock lockfile
support loses the lockfile content before it reaches the build host.

To rebuild and push the image after changing env/environment.yaml:

    # regenerate the lockfile (requires mamba/micromamba on PATH)
    uvx conda-lock lock -f env/environment.yaml -p linux-64 \
        --virtual-package-spec env/virtual-packages.yaml \
        --lockfile env/conda-lock.yml
    uvx conda-lock render -p linux-64 --kind explicit env/conda-lock.yml
    mv conda-linux-64.lock env/conda-linux-64.lock
    # regenerate the pip pin list (micromamba ignores "# pip" lock lines)
    python -c "
import re
raw = open('env/conda-lock.yml').read()
blocks = re.findall(r'^- name: ([^\n]+)\n  version: ([^\n]+)\n  manager: pip', raw, re.M)
open('env/requirements-locked.txt','w').write(
    '\n'.join(f'{n}=={v}' for n, v in sorted(set(blocks))) + '\n')
"
    # build and push (image must live in the same AWS account and region
    # that the Coiled workspace deploys into)
    docker build --platform linux/amd64 \
        -t 202533535508.dkr.ecr.us-east-1.amazonaws.com/aifs-demo:latest .
    aws ecr get-login-password --region us-east-1 | \
        docker login --username AWS --password-stdin 202533535508.dkr.ecr.us-east-1.amazonaws.com
    docker push 202533535508.dkr.ecr.us-east-1.amazonaws.com/aifs-demo:latest
"""

import coiled

coiled.create_software_environment(
    name="aifs-docker",
    container="202533535508.dkr.ecr.us-east-1.amazonaws.com/aifs-demo:latest",
)
