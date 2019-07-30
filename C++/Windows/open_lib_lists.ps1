subl
Set-Location $PSScriptRoot

$(Get-ChildItem -Path . -Filter CMakeLists.txt -Recurse).foreach({
    $name = $_.FullName
    if($name -match "(?<![eE]xample(s?))\\CMakeLists.txt"){
        Write-Host $name;
        subl $name;
    }
});
