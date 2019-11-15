#!/usr/bin/env pwsh

Write-Host
Write-Host "----------------------------------------------------------------------"

$build_dir         = 'C:\Source\Libraries'
$cmake_install_dir = 'C:\Source\CMake\Radicalware\Libraries\Projects'


$General_Purpose_Libs = 'General_Purpose_Libs'
$eXtended_STL         = 'eXtended_STL'
$Functional_STL       = 'Functional_STL'
$Modded_Objects       = 'Modded_Objects'

Write-Host "Copying Project Folders: "
Write-Host "1. $General_Purpose_Libs : Nexus, OS, SYS, Iterator, Timer"
Write-Host "2. $eXtended_STL         : xvector, xstring, xmap"
Write-Host "3. $Functional_STL       : re, ac, mc"
Write-Host "4. $Modded_Objects       : cc"

Copy-Item "$cmake_install_dir\Nexus\*"    $build_dir\$General_Purpose_Libs\Nexus\Project -Recurse -Force
Copy-Item "$cmake_install_dir\OS\*"       $build_dir\$General_Purpose_Libs\OS\Project -Recurse -Force
Copy-Item "$cmake_install_dir\SYS\*"      $build_dir\$General_Purpose_Libs\SYS\Project -Recurse -Force
Copy-Item "$cmake_install_dir\Iterator\*" $build_dir\$General_Purpose_Libs\Iterator\Project -Recurse -Force
Copy-Item "$cmake_install_dir\Timer\*"    $build_dir\$General_Purpose_Libs\Timer\Project -Recurse -Force

Copy-Item "$cmake_install_dir\xvector\*"  $build_dir\$eXtended_STL\xvector\Project -Recurse -Force 
Copy-Item "$cmake_install_dir\xstring\*"  $build_dir\$eXtended_STL\xstring\Project -Recurse -Force 
Copy-Item "$cmake_install_dir\xmap\*"     $build_dir\$eXtended_STL\xmap\Project -Recurse -Force

Copy-Item "$cmake_install_dir\ac\*"       $build_dir\$Functional_STL\ac\Project -Recurse -Force 
Copy-Item "$cmake_install_dir\re\*"       $build_dir\$Functional_STL\re\Project -Recurse -Force 
Copy-Item "$cmake_install_dir\mc\*"       $build_dir\$Functional_STL\mc\Project -Recurse -Force

Copy-Item "$cmake_install_dir\cc\*"       $build_dir\$Modded_Objects\cc\Project -Recurse -Force


Write-Host "All Lib Folders Updated!!"
Write-Host "----------------------------------------------------------------------`n"
