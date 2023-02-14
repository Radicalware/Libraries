
$TargetPath = "C:\Source\CMake\Radicalware\Libraries\Projects\HostStats\include"
Remove-Item -Recurse $TargetPath
New-Item -ItemType Directory $TargetPath
cd $TargetPath

foreach ( $item in $(Get-ChildItem C:\Source\CMake\Radicalware\Libraries\Projects\Stats\include))
{
    $NewName = $item.Name.Replace(".cuh", ".h")
    New-Item -ItemType HardLink -Name "$NewName" -Value $item.FullName
}


$TargetPath = "C:\Source\CMake\Radicalware\Libraries\Projects\HostStats\src"
Remove-Item -Recurse $TargetPath
New-Item -ItemType Directory $TargetPath
cd $TargetPath

foreach ( $item in $(Get-ChildItem C:\Source\CMake\Radicalware\Libraries\Projects\Stats\src))
{
    $NewName = $item.Name.Replace(".cu", ".cpp")
    New-Item -ItemType HardLink -Name "$NewName" -Value $item.FullName
}

cd "C:\Source\Radicalware\Libraries\GPU\Stats"

Copy-Item -Force .\FindHostStats.cmake C:\Source\CMake\Modules


