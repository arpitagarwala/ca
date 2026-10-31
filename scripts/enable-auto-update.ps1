$ErrorActionPreference = 'Stop'
$taskName = 'CA Bhaiya - Daily ICAI source refresh'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$nodePath = (Get-Command node.exe).Source
$scriptPath = Join-Path $projectRoot 'scripts\refresh.mjs'
$arguments = '-NoProfile -WindowStyle Hidden -Command "& ''' + $nodePath + ''' ''' + $scriptPath + '''"'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At '06:00'
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Refresh official ICAI material, validate published attempt applicability, update free full-text corpus, and deploy CA Bhaiya. Uses existing local Cloudflare login. No Cloudflare AI inference.' -Force | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName $taskName | Select-Object NextRunTime,LastTaskResult
