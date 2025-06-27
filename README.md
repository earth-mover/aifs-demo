# AIFS Forecasting Demo

> [!NOTE]
> This repo is for demonstration purposes only. It does not aspire to be a maintained package. If you want to build on top of it, fork this repo and modify it to your needs.

License: Apache 2.0

Some of the code has been adapted from an [ECWMF Notebook](https://huggingface.co/ecmwf/aifs-single-1.0) under the Apache 2.0 license.


## Usage

This code is packaged as a command-line script.

### ETL

```bash
% python main.py ingest --help
Usage: main.py ingest [OPTIONS] START_DATE END_DATE

Options:
  --repo-name TEXT
  --help            Show this message and exit.
```

### Forecast

```bash
% python main.py forecast --help
Usage: main.py forecast [OPTIONS] START_DATE END_DATE

Options:
  --ic-repo-name TEXT
  --target-repo-name TEXT
  --help                   Show this message and exit.
```
