Write-Host
Write-Host '--------------------------------------------------'
$cmake_build_dir = 'C:\source\include\'
$General_Purpose_Libs = 'General_Purpose_Libs'
$eXtended_STL = 'eXtended_STL'
$Modded_Objects = 'Modded_Objects'
$beta = 'beta'

Write-Host "Updating Find Files: "
Write-Host "1. $General_Purpose_Libs : OS, SYS, Iterator"
Write-Host "2. $eXtended_STL         : xvector, xstring, xmap"
Write-Host "3. $Modded_Objects       : cc"
Write-Host "4. $beta                 : vertex"

$find_cmake_path = 'C:\source\CMake\FindPackages\'

Copy-Item "$cmake_build_dir$General_Purpose_Libs\OS\FindOS.cmake"   $find_cmake_path -Force
Copy-Item "$cmake_build_dir$General_Purpose_Libs\SYS\FindSYS.cmake" $find_cmake_path -Force
Copy-Item "$cmake_build_dir$General_Purpose_Libs\Iterator\FindIterator.cmake"       $find_cmake_path -Force

Copy-Item "$cmake_build_dir$eXtended_STL\xvector\Findxvector.cmake" $find_cmake_path -Force 
Copy-Item "$cmake_build_dir$eXtended_STL\xstring\Findxstring.cmake" $find_cmake_path -Force 
Copy-Item "$cmake_build_dir$eXtended_STL\xmap\Findxmap.cmake"       $find_cmake_path -Force

Copy-Item "$cmake_build_dir$Modded_Objects\cc\Findcc.cmake"       $find_cmake_path -Force

Copy-Item "$cmake_build_dir$beta\vertex\Findvertex.cmake"       $find_cmake_path -Force


Write-Host "All Find Files Updated!!"
Write-Host "--------------------------------------------------`n"