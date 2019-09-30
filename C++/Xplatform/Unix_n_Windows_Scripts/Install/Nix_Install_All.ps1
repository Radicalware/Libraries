#!/usr/bin/env pwsh

param (
	[switch] $modify,   # you shouldn't use this, look at code if you really want to
	[switch] $lib,      # only install libs
	[switch] $examples  # only install examples
)

Set-Location "$PSScriptRoot"

#&".\clean.ps1"

$lib_installs = @()
$run_installs = @()

if(!$modify){
	$lib_installs = @(
		"../../functional_STL/ac/install.ps1",
		"../../functional_STL/mc/install.ps1",
		"../../functional_STL/re/install.ps1",

		"../../eXtended_STL/xstring/install.ps1",
		"../../eXtended_STL/xvector/install.ps1",
		"../../eXtended_STL/xmap/install.ps1",

		"../../General_Purpose_Libs/Iterator/install.ps1",
		"../../General_Purpose_Libs/OS/install.ps1",
		"../../General_Purpose_Libs/SYS/install.ps1",
		"../../General_Purpose_Libs/Nexus/install.ps1",

		"../../Modded_Objects/cc/install.ps1"
	)
		# --------------------------------------------------
	$run_installs = @(
		"../../functional_STL/ac/examples/run.ps1",
		"../../functional_STL/mc/examples/run.ps1",
		"../../functional_STL/re/examples/run.ps1",

		"../../eXtended_STL/xvector/examples/run.ps1",
		"../../eXtended_STL/xstring/example/run.ps1",
		"../../eXtended_STL/xmap/examples/run.ps1",


		"../../General_Purpose_Libs/OS/examples/run.ps1",
		"../../General_Purpose_Libs/SYS/examples/run.ps1",
		"../../General_Purpose_Libs/Nexus/examples/run.ps1",

		"../../Modded_Objects/cc/examples/run.ps1"
	)
};

# If you add a new lib, and only need to modify you can use the following code
# It is highly advised not to in most situations, you want to install
# dependencies in order.

if($modify){
	foreach($script in [string[]]("install.ps1","run.ps1")){
		$(Get-ChildItem -Path ../../ -Filter $script -Recurse).foreach({
			Write-Host "Installing: " $_.FullName
			&"$($_.FullName)" -Overwrite -No_Exec
		});
	};
}else{
	if($lib -eq $false -and $examples -eq $false){
		$lib = $true;
		$examples = $true;
	}

	if($lib -eq $true){
		foreach($install in $lib_installs){
			Set-Location "$PSScriptRoot"

			Write-Host $install
			&"$install" -Overwrite -No_Exec
		}
	}
	if($examples -eq $true){
		foreach($install in $run_installs){
			Set-Location "$PSScriptRoot"

			Write-Host $install
			&"$install" -Overwrite -No_Exec
		}
	}
}


Write-Host "`n`nAll Libs Installed!!"

