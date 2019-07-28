Write-Host
Write-Host '--------------------------------------------------'

$cmake_install_dir = 'C:\source\CMake\Radicalware\Libraries\cpp\code'


$General_Purpose_Libs = 'General_Purpose_Libs'
$eXtended_STL = 'eXtended_STL'
$Modded_Objects = 'Modded_Objects'
$beta = 'beta'


Write-Host "Copying Lib Folders: "
Write-Host "1. $General_Purpose_Libs : OS, SYS, Iterator"
Write-Host "2. $eXtended_STL         : xvector, xstring, xmap"
Write-Host "3. $Modded_Objects       : cc"
Write-Host "4. $beta                 : vertex"

Copy-Item "$cmake_install_dir\OS"  C:\source\include\$General_Purpose_Libs -Recurse -Force
Copy-Item "$cmake_install_dir\SYS" C:\source\include\$General_Purpose_Libs -Recurse -Force
Copy-Item "$cmake_install_dir\Iterator" C:\source\include\$General_Purpose_Libs -Recurse -Force

Copy-Item "$cmake_install_dir\xvector" C:\source\include\$eXtended_STL -Recurse -Force 
Copy-Item "$cmake_install_dir\xstring" C:\source\include\$eXtended_STL -Recurse -Force 
Copy-Item "$cmake_install_dir\xmap"    C:\source\include\$eXtended_STL -Recurse -Force

Copy-Item "$cmake_install_dir\cc"    C:\source\include\$Modded_Objects -Recurse -Force

Copy-Item "$cmake_install_dir\vertex"    C:\source\include\$beta -Recurse -Force


Write-Host "All Lib Folders Updated!!"
Write-Host "--------------------------------------------------`n"