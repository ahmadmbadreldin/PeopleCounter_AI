$ErrorActionPreference = 'Stop'

$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $MyInvocation.MyCommand.Path))
$taskName = 'ColorsBeauty-PeopleCounter'
$scriptNames = @(
    'people_counter_hikvision.py',
    'watchdog.py',
    'setup_branch.py',
    'staff_enrollment.py'
)

Write-Host 'Stopping the People Counter scheduled task...'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $task) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}

# Match both the project path and one of our exact Python entry-point names.
# This deliberately leaves unrelated python.exe/pythonw.exe processes alone.
$escapedRoot = [Regex]::Escape($projectRoot.TrimEnd('\'))
$escapedScripts = ($scriptNames | ForEach-Object { [Regex]::Escape($_) }) -join '|'
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('python.exe', 'pythonw.exe') -and
    $_.CommandLine -and
    $_.CommandLine -match $escapedRoot -and
    $_.CommandLine -match "($escapedScripts)"
}

foreach ($process in $targets) {
    Write-Host ("Stopping PID {0}: {1}" -f $process.ProcessId, $process.Name)
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Milliseconds 750

$remaining = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('python.exe', 'pythonw.exe') -and
    $_.CommandLine -and
    $_.CommandLine -match $escapedRoot -and
    $_.CommandLine -match "($escapedScripts)"
}

if ($remaining) {
    Write-Host 'Some People Counter processes could not be stopped.' -ForegroundColor Red
    exit 1
}

Write-Host 'All old People Counter processes are stopped.' -ForegroundColor Green
Write-Host 'The automatic-start task is still installed, but is stopped for this Windows session.'
exit 0
