$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

uv python install 3.11
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    uv venv --python 3.11 .venv
}

$python = ".\.venv\Scripts\python.exe"

# Install the proven CUDA 12.4 PyTorch build first. Installing OpenCLIP afterward
# lets its dependency resolver reuse this GPU build instead of downloading a CPU
# torch wheel and replacing it again.
uv pip install --python $python `
    "torch==2.6.0" `
    "torchvision==0.21.0" `
    --index-url https://download.pytorch.org/whl/cu124

uv pip install --python $python `
    "open_clip_torch==3.3.0" `
    "fastapi==0.136.3" `
    "uvicorn==0.32.1" `
    "pillow>=11.0,<12" `
    "requests>=2.32,<3"

& $python -c "import torch, open_clip; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('OpenCLIP:', open_clip.__version__)"

Write-Host "Semantic ranker setup complete. Start it with .\start.ps1"
