

$ProjectPath = "C:\Source\CMake\Radicalware\Libraries\Projects\HostStats"

if($(Test-Path $ProjectPath) -eq $false)
{
    New-Item -ItemType Directory $ProjectPath
}

$TargetPath = "$ProjectPath\include"
Remove-Item -Recurse $TargetPath
New-Item -ItemType Directory $TargetPath
Set-Location $TargetPath

foreach ( $item in $(Get-ChildItem C:\Source\CMake\Radicalware\Libraries\Projects\Stats\include))
{
    $NewName = $item.Name.Replace(".cuh", ".h")
    New-Item -ItemType HardLink -Name "$NewName" -Value $item.FullName
}


$TargetPath = "$ProjectPath\src"
Remove-Item -Recurse $TargetPath
New-Item -ItemType Directory $TargetPath
Set-Location $TargetPath

foreach ( $item in $(Get-ChildItem C:\Source\CMake\Radicalware\Libraries\Projects\Stats\src))
{
    $NewName = $item.Name.Replace(".cu", ".cpp")
    New-Item -ItemType HardLink -Name "$NewName" -Value $item.FullName
}

Set-Location "C:\Source\Radicalware\Libraries\GPU\Stats"

Copy-Item -Force .\FindHostStats.cmake C:\Source\CMake\Modules


