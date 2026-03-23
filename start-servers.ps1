# voice-lab server launcher -- run from admin or regular PowerShell on Sunny
# Usage: .\start-servers.ps1
# Starts STT (faster-whisper, :9001) and TTS (piper, :9002)

$VL = "C:\Users\kodep\voice-lab"

# Kill any existing instances
& "$VL\stop-servers.ps1"

Write-Host 'Starting STT server (faster-whisper) on :9001...'
Start-Process -FilePath python -ArgumentList "$VL\servers\stt_server.py","--port","9001","--device","cuda" -RedirectStandardOutput "$VL\stt.log" -RedirectStandardError "$VL\stt_err.log" -WindowStyle Hidden

Write-Host 'Starting TTS server (piper) on :9002...'
Start-Process -FilePath python -ArgumentList "$VL\servers\tts_server.py","--port","9002","--voice","$VL\models\en_US-lessac-medium.onnx" -RedirectStandardOutput "$VL\tts.log" -RedirectStandardError "$VL\tts_err.log" -WindowStyle Hidden

Start-Sleep -Seconds 3

# Health check
try { $stt = Invoke-RestMethod http://localhost:9001/health } catch { $stt = @{status='FAILED'} }
try { $tts = Invoke-RestMethod http://localhost:9002/health } catch { $tts = @{status='FAILED'} }

Write-Host "STT: $($stt.status)  TTS: $($tts.status)"
