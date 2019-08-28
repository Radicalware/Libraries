#!/usr/bin/env pwsh

Set-Location $PSScriptRoot

$rex = ""
if($global:PSVersionTable.Platform -eq "Unix"){
	$rex = "^.*ex_[\w_\d]+([^(\.exe)]?)$";
}else{
	$rex = "^.*ex_[\w_\d]+\.exe$";
}

$counter = $(Get-ChildItem -Path ../../ -Filter run.ps1 -Recurse).Count

$executed = New-Object System.Collections.ArrayList

$(Get-ChildItem -Path ../../ -File -Recurse).foreach({
    $name = $_.FullName
    if($name -match $rex -and $name -notmatch "^.*ex_SYS.*$")
	{
		$executed.Add($name) | Out-Null

		$out = "$((Invoke-Expression "$($name) 2>&1" ) | Out-String)"
		Write-Host "$($("|" * 130) + $("`n") + $name + $("`n") + $out )"
		$counter--
    }
	elseif($name -match $rex -and $name -match "^.*ex_SYS.*$")
	{
		$executed.Add($name) | Out-Null

		$test_subject = "$($([regex]::replace($name, "Build.*$", `"`"))+$("execute.ps1"))";
		$out = "$((Invoke-Expression "$($test_subject) 2>&1" ) | Out-String)"
		Write-Host "$($("|" * 130) + $("`n") + $test_subject + $("`n") + $out )"
		$counter--;
	}
});


Set-Location $PSScriptRoot
$exe_path_count = $(Get-ChildItem -Path ../../ -File -Filter run.ps1 -Recurse).Length

if($executed.Count -ne $exe_path_count)
{
	write-host "Path Count: "$exe_path_count
	write-host "Executed  : "$executed.Count

	foreach($exe in $executed){
		Write-Host $exe;
	}
}else{
	Write-Host "All Scripts Ran!!";
}


