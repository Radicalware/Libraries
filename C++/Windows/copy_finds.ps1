Write-Host
Write-Host '-------------------------------------------------------------------------'

Write-Host "Updating Find Files: "
$find_cmake_path = 'C:\source\CMake\FindPackages\'

$(Get-ChildItem -Path . -Recurse -Force).foreach({ # -Force to find hidden files                
    $name = $_.FullName                                                                         
    if($name -match "^.*\\(Find[\w\d_-]+\.cmake)$"){                    
        if(Test-Path -Path $name){                                                              
            Write-Host $name;      
            Copy-Item $name $find_cmake_path -Force                                                             
        }                                                                                       
    }                                                                                           
});                                                                                             


Write-Host "All Find Files Updated!!"
Write-Host '-------------------------------------------------------------------------'
Write-Host

