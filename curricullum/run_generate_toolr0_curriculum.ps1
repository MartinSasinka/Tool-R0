# Generate synthetic Tool-R0 curriculum (Windows)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$envFile = Join-Path (Get-Location) ".env"
if (-not $env:OPENROUTER_API_KEY -and (Test-Path $envFile)) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*OPENROUTER_API_KEY\s*=\s*(.+)\s*$') {
            $env:OPENROUTER_API_KEY = $matches[1].Trim().Trim('"').Trim("'")
        }
    }
}

if (-not $env:OPENROUTER_API_KEY) {
    Write-Error "OPENROUTER_API_KEY is not set (env or .env)"
}

if (-not $env:MODEL) { $env:MODEL = "deepseek/deepseek-v4-flash" }
if (-not $env:N_FINAL) { $env:N_FINAL = "500" }
if (-not $env:N_GENERATE) { $env:N_GENERATE = "1500" }
if (-not $env:MAX_STAGES) { $env:MAX_STAGES = "3" }
if (-not $env:PARALLEL_WORKERS) { $env:PARALLEL_WORKERS = "16" }
if (-not $env:USE_EXECUTOR) { $env:USE_EXECUTOR = "1" }

python curricullum/run_generate_toolr0_curriculum.py
