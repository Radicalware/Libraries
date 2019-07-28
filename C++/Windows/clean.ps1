

Set-Location $PSScriptRoot

$(Get-ChildItem -Path . -Recurse -Force).foreach({ # -Force to find hidden files
    $name = $_.FullName    
    if($_.PSIsContainer -and $name -match "^.*\\(Release|Debug|out|\.vs)$"){
        if(Test-Path -Path $name){
            Write-Host $name;
            Remove-Item $_.FullName -Recurse -Force;
        }
    }
});