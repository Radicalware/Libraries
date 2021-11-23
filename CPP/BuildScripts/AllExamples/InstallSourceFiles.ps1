

$GeneralPurposeLibs = 'GeneralPurposeLibs'
$ExtendedSTL         = 'ExtendedSTL'
$FunctionalSTL       = 'FunctionalSTL'
$ModdedObjects       = 'ModdedObjects'

Write-Host "Copying Project Folders: "
Write-Host "1. GeneralPurposeLibs :  Macros, Nexus, Socket, OS"
Write-Host "                          SYS, Iterator, Timer, Date"
Write-Host "                          CudaBridge, Stash, JSON, AES"
Write-Host "2. ExtendedSTL         : Memory, xvector, xstring, xmap"
Write-Host "3. FunctionalSTL       : re, ac, mc"
Write-Host "4. ModdedObjects       : cc"

class TheHandler
{
    $BaseFolder;
    $InstallDir;

    $ProjectFolder;

    TheHandler()
    {
        $this.BaseFolder = "$PSScriptRoot\..\.."
        $this.InstallDir = "C:\Source\CMake\Radicalware\Libraries\Projects\"
    }

    [void] CopyProject([string] $ProjctName)
    {
        $Destination = "$($this.InstallDir)\$ProjctName"
        if($(Test-Path $Destination) -eq $false){
            mkdir $Destination | Out-Null
        }

        Copy-Item -Recurse -Force "$($this.BaseFolder)\$($this.ProjectFolder)\$ProjctName\Project\*" $Destination
    }
};



$Handler = [TheHandler]::new()

$Handler.ProjectFolder = $GeneralPurposeLibs
$Handler.CopyProject("Macros")
$Handler.CopyProject("Nexus")
$Handler.CopyProject("Socket")
$Handler.CopyProject("OS")
$Handler.CopyProject("SYS")
$Handler.CopyProject("Iterator")
$Handler.CopyProject("Timer")
$Handler.CopyProject("Date")
$Handler.CopyProject("JSON")
$Handler.CopyProject("Stash")
$Handler.CopyProject("AES")
$Handler.CopyProject("CudaBridge")

$Handler.ProjectFolder = $ExtendedSTL
$Handler.CopyProject("Memory")
$Handler.CopyProject("xvector")
$Handler.CopyProject("xstring")
$Handler.CopyProject("xmap")

$Handler.ProjectFolder = $FunctionalSTL
$Handler.CopyProject("RE")
$Handler.CopyProject("AC")
$Handler.CopyProject("MC")

$Handler.ProjectFolder = $ModdedObjects
$Handler.CopyProject("CC")


