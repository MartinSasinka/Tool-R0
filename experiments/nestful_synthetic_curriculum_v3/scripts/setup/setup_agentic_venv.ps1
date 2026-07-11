# Create and populate the agentic data-generation venv (Windows).
# Usage (repo root):
#   powershell -ExecutionPolicy Bypass -File experiments/nestful_synthetic_curriculum_v3/scripts/setup/setup_agentic_venv.ps1
param(
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu124"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$V3Root = Join-Path $RepoRoot "experiments\nestful_synthetic_curriculum_v3"
$VenvDir = Join-Path $V3Root ".venv"
$ReqFile = Join-Path $V3Root "requirements-agentic.txt"
$Py = Join-Path $VenvDir "Scripts\python.exe"
$Pip = Join-Path $VenvDir "Scripts\pip.exe"

Write-Host "=== agentic venv setup ==="
Write-Host "repo : $RepoRoot"
Write-Host "venv : $VenvDir"

if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}

& $Py -m pip install --upgrade pip wheel setuptools
& $Pip install torch --index-url $CudaIndex
& $Pip install -r $ReqFile

Write-Host ""
Write-Host "--- verify ---"
& $Py -c @"
import importlib
mods = ['torch', 'transformers', 'bitsandbytes', 'accelerate', 'pytest', 'dotenv']
for m in mods:
    try:
        importlib.import_module(m)
        print(f'  {m}: OK')
    except Exception as exc:
        print(f'  {m}: FAIL ({exc})')
import torch
print(f'  cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  gpu: {torch.cuda.get_device_name(0)}')
"@

Write-Host ""
Write-Host "Activate:"
Write-Host "  .\experiments\nestful_synthetic_curriculum_v3\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Load API key from repo .env (optional):"
Write-Host '  Get-Content .env | ForEach-Object { if ($_ -match ''^([^#=]+)=(.*)$'') { Set-Item -Path "env:$($matches[1].Trim())" -Value $matches[2].Trim().Trim(''"'') } }'
Write-Host ""
Write-Host "Pilot (hybrid):"
Write-Host '  $env:WEAK_SOLVER_BACKEND="local"; $env:LOCAL_WEAK_4BIT="1"; python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_curriculum_v4_agentic_openrouter.py --pilot --stages stage2_2call_agentic_openrouter --seed 44'
