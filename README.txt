CloudCast reproducible analysis
===============================

Purpose
-------
This project forecasts Microsoft Azure LLM generated-token workload. The two
raw request traces are aggregated into one uniformly sampled observation per
minute for the common period in which both services are present.

Recommended environment
-----------------------
Use 64-bit Python 3.12. The tested setup uses uv. From the CloudCast project
directory:

    $env:UV_CACHE_DIR = (Join-Path (Get-Location) '.uv-cache')
    uv venv --python 3.12.12 .venv
    uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt

The scripts can then be run without activating the environment:

    .\.venv\Scripts\python.exe scripts\01_build_dataset.py

If Python 3.12 and the Windows py launcher are installed, a standard venv is
also supported:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Raw files
---------
Place these files in data\raw:

    AzureLLMInferenceTrace_code_1week.csv
    AzureLLMInferenceTrace_conv_1week.csv

Run order
---------
Run every command from the CloudCast project directory.

1. Build the canonical one-minute dataset:

    .\.venv\Scripts\python.exe scripts\01_build_dataset.py

   The script streams the large files in chunks. It writes:

    data\processed\azure_llm_one_minute.csv
    data\processed\azure_llm_one_minute_metadata.json

   To deliberately rebuild an existing processed file:

    .\.venv\Scripts\python.exe scripts\01_build_dataset.py --force

2. Generate the dataset-description analysis:

    .\.venv\Scripts\python.exe scripts\02_describe_dataset.py

   Results are written to outputs\dataset_description.

3. Generate the stationarity analysis:

    .\.venv\Scripts\python.exe scripts\03_stationarity.py

   Results are written to outputs\stationarity.

4. Generate additive and multiplicative STL results:

    .\.venv\Scripts\python.exe scripts\04_decomposition.py

   Results are written to outputs\decomposition.

Important conventions
---------------------
- Time is stored in UTC.
- Sampling frequency is one minute.
- The dependent variable is total_generated_tokens.
- The train/test split is chronological: first 80% for training, final 20%
  for testing. With 7,200 observations this gives 5,760 training rows and
  1,440 test rows.
- The selected stationarity transformation is:

    log1p(total_generated_tokens).diff()

- Scripts fail when required columns, timestamps, derived totals, or uniform
  one-minute spacing are invalid.

Troubleshooting
---------------
If the command "python" points to an uninstalled interpreter, reinstall
64-bit Python 3.12 or invoke the Python executable inside .venv directly.

