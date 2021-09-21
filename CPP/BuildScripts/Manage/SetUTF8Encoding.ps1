#!/usr/bin/env pwsh

write-host "Target: "$args[0]
Set-Content $args[0] -Encoding utf8 -Value $(Get-Content $args[0]);
