import coiled

# Create software environment
coiled.create_software_environment(
    name="aifs-conda",
    conda="environment.yaml",
    pip="requirements.txt",
    gpu_enabled=True,
    force_rebuild=True,
)