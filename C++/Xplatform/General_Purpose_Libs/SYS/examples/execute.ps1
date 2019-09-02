#!/usr/bin/env pwsh

Set-Location "$PSScriptRoot"

$is_unix =  $($global:PSVersionTable.Platform -eq "Unix");

if($is_unix){
	./Build/Nix/Release/Release/ex_SYS --key-A sub-A-1 sub-A-2 sub-A-3 --key-B sub-B-1 sub-B-2 -a -bcdp 8080  9090 -ef -g 
}else{
	.\Build\Windows\Release\Release\bin\ex_SYS.exe --key-A sub-A-1 sub-A-2 sub-A-3 --key-B sub-B-1 sub-B-2 -a -bcdp 8080  9090 -ef -g 
}
