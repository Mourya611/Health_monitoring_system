$ErrorActionPreference = "Stop"

$services = @(
    @{ Name = "frontend"; Url = "http://127.0.0.1:5173" },
    @{ Name = "backend"; Url = "http://127.0.0.1:8000/health" },
    @{ Name = "nlp"; Url = "http://127.0.0.1:8001/health" }
)

$failed = $false

foreach ($svc in $services) {
    try {
        $resp = Invoke-WebRequest -Uri $svc.Url -UseBasicParsing -TimeoutSec 5
        Write-Host ("[OK]   {0} -> {1}" -f $svc.Name, $resp.StatusCode)
    } catch {
        $failed = $true
        Write-Host ("[DOWN] {0} -> {1}" -f $svc.Name, $_.Exception.Message)
    }
}

if ($failed) {
    exit 1
}

exit 0
