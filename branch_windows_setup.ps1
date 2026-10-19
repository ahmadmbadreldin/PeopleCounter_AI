$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$watchdog = Join-Path $projectRoot 'watchdog.py'
$systemCheck = Join-Path $projectRoot 'system_check.py'
$config = Join-Path $projectRoot 'branch_config.json'
$taskName = 'ColorsBeauty-PeopleCounter'

if (-not (Test-Path -LiteralPath $pythonw)) {
    throw 'Run "1 - Configure Branch.cmd" once first so the local Python environment is installed.'
}
if (-not (Test-Path -LiteralPath $config)) {
    throw 'Configure and save the branch first using "1 - Configure Branch.cmd".'
}

Write-Host 'Running the production system check before changing Windows settings...'
& $python $systemCheck
if ($LASTEXITCODE -ne 0) {
    throw 'Production system check failed. Fix every [FAIL] item before installing startup.'
}

$action = New-ScheduledTaskAction `
    -Execute $pythonw `
    -Argument ('"' + $watchdog + '"') `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description 'Starts the branch People Counter watchdog after Windows sign-in.' `
    -Force | Out-Null

# The branch computer may still be shut down normally every night.
# Monitor timeout is intentionally unchanged; the display may turn off while processing continues.
powercfg.exe /change standby-timeout-ac 0
if ($LASTEXITCODE -ne 0) { throw 'Unable to disable Sleep on AC power.' }
powercfg.exe /change hibernate-timeout-ac 0
if ($LASTEXITCODE -ne 0) { throw 'Unable to disable Hibernate on AC power.' }
powercfg.exe /change standby-timeout-dc 0
if ($LASTEXITCODE -ne 0) { throw 'Unable to disable Sleep on battery power.' }
powercfg.exe /change hibernate-timeout-dc 0
if ($LASTEXITCODE -ne 0) { throw 'Unable to disable Hibernate on battery power.' }

Write-Host ''
Write-Host 'Branch PC setup completed successfully.' -ForegroundColor Green
Write-Host '- People Counter will start after this Windows user signs in.'
Write-Host '- Watchdog will restart it after a crash or frozen heartbeat.'
Write-Host '- Sleep and Hibernate are disabled on AC and battery power.'
Write-Host '- Normal nightly Windows shutdown remains available.'
