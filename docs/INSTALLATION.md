# One Python environment

The application, RAG, agents, tests and computer vision use **`.venv`**. Install
the `dev` and `vision` extras into this environment. Pytest never installs
dependencies, starts services or downloads checkpoints/models.

## Windows with the current NVIDIA GPU

Create the environment with a compatible Python (the validated workstation uses
Python 3.14.7), then use only its interpreter and pip:

```powershell
python -m venv .venv
# Explicitly select the validated CUDA wheels for this NVIDIA workstation:
.\.venv\Scripts\python.exe -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu132 'torch==2.14.1+cu132' 'torchvision==0.29.1+cu132'
.\.venv\Scripts\python.exe -m pip install -e ".[dev,vision]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
.\.venv\Scripts\Activate.ps1
```

The validated pair is torch **2.14.1+cu132** / torchvision **0.29.1+cu132**,
with Ultralytics **8.4.176** and CUDA runtime **13.2**. Choose a PyTorch build
compatible with the driver/hardware on another workstation using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The project extra has portable version constraints and does not force a CUDA
index or NVIDIA-only package. An installed CUDA build satisfying these constraints
does not need to be replaced when installing the project extra.

## CPU-only Windows or Linux

CUDA is optional. For a new CPU environment, select the CPU wheels first, then
install the application and extras into the activated environment:

```text
python -m venv .venv
# Activate .venv (Scripts/Activate.ps1 on Windows; source .venv/bin/activate on Linux).
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1" "torchvision==0.29.1"
python -m pip install -e ".[dev,vision]"
python -m pip check
```

On Linux with an NVIDIA GPU, choose the compatible CUDA index explicitly instead
of the CPU index. The command above is for a CPU installation, not a migration
of a working CUDA workstation.

## macOS

Create and activate `.venv`, then install `.[dev,vision]` from the ordinary package
index. Do not select the CUDA wheel index on macOS. PyTorch uses the supported
macOS backend/CPU; NVIDIA/CUDA availability is not a project prerequisite.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,vision]'
python -m pip check
```

Linux/macOS installation instructions describe the supported configuration;
the migration and real GPU tests in this report were executed only on Windows.

## Devices, checkpoints and pytest

`pytest` runs all tests. Use `pytest -m unit`, `postgres`, `qdrant`, `rag` or
`vision` for a selection. Vision runs directly in this same interpreter; there
is no delegation and no `VISION_PYTHON` setting. The checkpoint integration can
run on CPU; only an explicit GPU request requires CUDA.

Training configurations record the historical GPU baseline. On a machine without
CUDA, select `--device cpu` explicitly for the training/evaluation scripts; for
inference use `DAMAGE_DEVICE=cpu` and `PART_DEVICE=cpu` when necessary. Changing
devices does not authorize restarting a completed baseline or overwriting its
metrics/checkpoints.

Provide the existing local damage and vehicle-parts checkpoints. Pytest and the
inference adapters never download missing weights. RAG needs its small existing
corpus and the deliberately preloaded embedding cache described in the README.

Start the test containers manually when needed:

```text
docker compose --profile test up -d --no-deps --wait postgres-test qdrant-test
pytest
```

The former `.venv-vision` is retained as a fallback during migration. Once the
single-environment validation succeeds it is unnecessary; its removal remains
the user's decision. The separate optional CarDD/FiftyOne data-preparation
environment remains independent because FiftyOne does not support Python 3.14;
it is not used for application, model inference or pytest execution.
