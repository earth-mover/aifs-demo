import coiled


# a big fat GPU-enabled environment
coiled.create_software_environment(
    name="aifs-conda",
    conda="env/environment.yaml",
    pip="env/requirements.txt",
    gpu_enabled=True,
)


# lightweight environment for ETL tasks
coiled.create_software_environment(
    name="aifs-etl",
    pip="env/requirements-etl.txt",
)
