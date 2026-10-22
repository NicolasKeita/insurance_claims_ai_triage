# One Python environment

Use **Python 3.12** for the application, RAG, agents, tests, computer vision and
FiftyOne data preparation in **`.venv`**. The default is recorded in
`.python-version`. Install the `dev`, `vision` and `cardd-hf` extras together.
Pytest never installs dependencies, starts services or downloads checkpoints/models.

## Windows with the current NVIDIA GPU

Create the environment explicitly with Python 3.12, then use only its interpreter
and pip. If the Python launcher is unavailable, use the absolute path of your
Python 3.12 interpreter instead of `py -3.12`:

```powershell
py -3.12 -m venv .venv
# Explicitly select the validated CUDA wheels for this NVIDIA workstation:
.\.venv\Scripts\python.exe -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu132 'torch==2.14.1+cu132' 'torchvision==0.29.1+cu132'
.\.venv\Scripts\python.exe -m pip install -e ".[dev,vision,cardd-hf]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
.\.venv\Scripts\Activate.ps1
```

The validated pair is torch **2.14.1+cu132** / torchvision **0.29.1+cu132**,
with Python **3.12.14**, Ultralytics **8.4.176** and CUDA runtime **13.2**.
FiftyOne **1.22.1** shares this environment; its Starlette constraint resolves
to **1.3.1**, which is also compatible with FastAPI **0.141.1**.
Choose a PyTorch build
compatible with the driver/hardware on another workstation using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The project extra has portable version constraints and does not force a CUDA
index or NVIDIA-only package. An installed CUDA build satisfying these constraints
does not need to be replaced when installing the project extra.

## CPU-only Windows or Linux

CUDA is optional. For a new CPU environment, select the CPU wheels first, then
install the application and extras into the activated environment:

```text
python3.12 -m venv .venv
# On Windows use: py -3.12 -m venv .venv
# Activate .venv (Scripts/Activate.ps1 on Windows; source .venv/bin/activate on Linux).
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1" "torchvision==0.29.1"
python -m pip install -e ".[dev,vision,cardd-hf]"
python -m pip check
```

On Linux with an NVIDIA GPU, choose the compatible CUDA index explicitly instead
of the CPU index. The command above is for a CPU installation, not a migration
of a working CUDA workstation.

## macOS

Create and activate a Python 3.12 `.venv`, then install `.[dev,vision,cardd-hf]`
from the ordinary package index. Do not select the CUDA wheel index on macOS. PyTorch uses the supported
macOS backend/CPU; NVIDIA/CUDA availability is not a project prerequisite.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,vision,cardd-hf]'
python -m pip check
```

Linux/macOS installation instructions describe the supported configuration;
the migration and real GPU tests in this report were executed only on Windows.

## Devices, checkpoints and pytest

Use `pytest --ignore=tests/e2e` for tests outside the end-to-end suite, or add
`-m unit`, `postgres`, `qdrant`, `rag` or `vision` for a selection.
Agents must not run end-to-end tests without an explicit user request; see
`AGENTS.md`. The `unit` marker alone does not exclude end-to-end tests.
Vision runs directly in this same interpreter; there
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
pytest --ignore=tests/e2e
```

Python 3.12 supports the application's dependencies and the optional FiftyOne
data-preparation stack in the same interpreter. Neither `.venv-vision` nor
`.venv-cardd` is needed once the combined environment has passed validation.
When migrating, preserve the previous environments until `pip check`, unit
tests outside `tests/e2e`, real GPU inference and FiftyOne imports succeed.
Recreate `.venv` at its final path rather than relocating a virtual environment:
installed command launchers embed absolute interpreter paths.
