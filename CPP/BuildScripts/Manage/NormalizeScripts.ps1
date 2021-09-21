#!/usr/bin/env pwsh


# Run this from a Unix box if you want to remove \r\n
$(Get-ChildItem $PSScriptRoot/../.. -Recurse ).where({ $([regex]::Matches($_.Name, "^.*\.(h|cpp|ps1|cmake)$"))}).ForEach({Set-Content $_.FullName -Encoding utf8 -Value $(Get-Content $_.FullName); });




# Set-Content Normalize_Scripts.ps1 -Encoding utf8 -Value $(Get-Content Normalize_Scripts.ps1);
