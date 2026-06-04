param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..")
$Python = ".\.venv\Scripts\python.exe"
$PythonDisplay = Join-Path $RootDir ".venv\Scripts\python.exe"
$Database = Join-Path $RootDir "data\geo_datasets\geo_answers.sqlite"
$Url = "http://127.0.0.1:$Port/"

function Wait-ForDashboard {
    param([string]$TargetUrl)

    for ($i = 0; $i -lt 30; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $TargetUrl -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

Set-Location $RootDir

if (-not (Test-Path $Python)) {
    Write-Host "Project virtual environment was not found:" -ForegroundColor Red
    Write-Host "  $PythonDisplay"
    Write-Host "Create it first, then install requirements with:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path $Database)) {
    Write-Host "SQLite database was not found:" -ForegroundColor Yellow
    Write-Host "  $Database"
    Write-Host "The dashboard can start, but SQLite-backed views will be empty until data is imported."
}

if (Wait-ForDashboard -TargetUrl $Url) {
    Write-Host "Dashboard service is already running." -ForegroundColor Yellow
    Write-Host "URL: $Url"
    if (-not $NoBrowser) {
        Start-Process $Url
    }
    exit 0
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing) {
    Write-Host "Port $Port is already in use, but the dashboard did not respond at $Url." -ForegroundColor Red
    Write-Host "Stop the process using this port, or run with another port, for example:"
    Write-Host "  .\start_dashboard.bat -Port 8001"
    Read-Host "Press Enter to exit"
    exit 1
}

try {
    & $Python --version | Out-Null
} catch {
    Write-Host "Project virtual environment exists but could not start Python:" -ForegroundColor Red
    Write-Host "  $PythonDisplay"
    Write-Host "Recreate the virtual environment and reinstall requirements, then run this script again."
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting GEO Search dashboard..." -ForegroundColor Green
Write-Host "Backend: uvicorn backend.app.main:app"
Write-Host "Frontend: static dashboard served by FastAPI"
Write-Host "URL: $Url"
Write-Host ""

$arguments = @("-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Reload) {
    $arguments += "--reload"
}

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($TargetUrl)
        for ($i = 0; $i -lt 30; $i++) {
            try {
                $response = Invoke-WebRequest -Uri $TargetUrl -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                    Start-Process $TargetUrl
                    return
                }
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } -ArgumentList $Url | Out-Null
}

& $Python @arguments
