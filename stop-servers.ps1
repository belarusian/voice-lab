# voice-lab server shutdown
# Usage: .\stop-servers.ps1

# Get-Process.CommandLine is unreliable -- use CIM which always has it
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'stt_server|tts_server'
}

if ($procs) {
    $procs | ForEach-Object {
        Write-Host "Killing PID $($_.ProcessId): $($_.CommandLine)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
} else {
    Write-Host 'No voice-lab servers found.'
}

# Fallback: kill anything holding ports 9001/9002
foreach ($port in 9001, 9002) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $pid = $conn.OwningProcess
        Write-Host "Port $port still held by PID $pid -- killing"
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
}

Write-Host 'Servers stopped.'
