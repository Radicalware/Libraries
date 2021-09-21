#!/usr/bin/env pwsh

$library_dir = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $library_dir = "/opt/Radicalware/Libraries"
}else{
    $library_dir = "C:\Source\CMake\Radicalware\Libraries"
}

$header_files_path = $library_dir  + "/Headers/"
$ProjectFiles_path = $library_dir + "/Projects/"

$(Get-ChildItem -Path $ProjectFiles_path -File -Recurse).foreach({
    $name = $_.FullName
    if($name -match "^.*\.h$" -and $name -notmatch "^.*CMakeFiles.*$"){
        $project_file_path = $($name | Select-String -Pattern "(?<=include[\\/])(.*)").Matches.Groups[0].Value
        $new_path = $header_files_path + $project_file_path
        $create_dir = $($new_path | Select-String -Pattern "^.*[\\/]").Matches.Groups[0].Value
        
        If (!(test-path $create_dir))
        {
            New-Item -ItemType Directory -Path $create_dir | Out-Null
        }
        New-Item -ItemType SymbolicLink -Path $new_path -Value $name
        Write-Host $project_file_path
    }
});
