$ErrorActionPreference = 'Stop'
$taskName = 'ColorsBeauty-PeopleCounter'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $task) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host 'Automatic People Counter startup was removed.' -ForegroundColor Green
} else {
    Write-Host 'No People Counter startup task was found.'
}
Write-Host 'Sleep/Hibernate values were not changed by removal; set your preferred values in Windows Power settings.'
