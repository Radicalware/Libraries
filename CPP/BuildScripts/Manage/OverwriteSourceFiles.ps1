#!/usr/bin/env pwsh

Write-Host
Write-Host "----------------------------------------------------------------------"

# ALERT!!! AFTER MAKING MODS, BE SURE TO COPY THIS FILE TO YOUR POWRSHELL PATH !!!


$GeneralPurposeLibs  = 'GeneralPurposeLibs'
$ExtendedSTL         = 'ExtendedSTL'
$FunctionalSTL       = 'FunctionalSTL'
$ModdedObjects       = 'ModdedObjects'

Write-Host "Copying Project Folders: "
Write-Host "1. GeneralPurposeLibs :  Macros, Nexus, Socket, OS"
Write-Host "                          SYS, Iterator, Timer, Date"
Write-Host "                          CudaBridge, Stash, JSON"
Write-Host "2. ExtendedSTL         : xvector, xstring, xmap"
Write-Host "3. FunctionalSTL       : re, ac, mc"
Write-Host "4. ModdedObjects       : cc"

# ALERT!!! AFTER MAKING MODS, BE SURE TO COPY THIS FILE TO YOUR POWRSHELL PATH !!!

class Lib_Handler
{
    [string] $Dir;
    
    [string] $GitSourceDir    = "C:\Source\Radicalware\Libraries"

    [string] $CMakeInstallDir = 'C:\Source\CMake\Radicalware\Libraries\Projects'
    [string] $CMakeHeaderDir  = 'C:\Source\CMake\Radicalware\Libraries\Headers'

    Lib_Handler()
    {
    }

    [void] CopyLib([string] $Lib)
    {    
        $From = "$($this.CMakeInstallDir)\$Lib\*"
        $To   = "$($this.GitSourceDir)\$($this.dir)\$Lib\Project"
        # Write-Host "Copying: $From >> $To"

        if($(Test-Path $To) -eq $false){
            mkdir $To
        }

        try{        
            if($(Test-Path $From) -eq $false){
                Write-Host "Folder Not Found: ${From}"
            }else{
                Copy-Item -Recurse -Force $From $To
            }
        }catch
        {
            Write-Host "Can't Copy: $From >> $To"
        }
    }
}

$handler = [Lib_Handler]::new()

$handler.dir = $GeneralPurposeLibs
$handler.CopyLib("Macros")
$handler.CopyLib("Nexus")
$handler.CopyLib("Socket")
$handler.CopyLib("OS")
$handler.CopyLib("SYS")
$handler.CopyLib("Iterator")
$handler.CopyLib("Timer")
$handler.CopyLib("Date")
$handler.CopyLib("JSON")
$handler.CopyLib("Stash")
$handler.CopyLib("CudaBridge")

$handler.dir = $ExtendedSTL
$handler.CopyLib("xvector")
$handler.CopyLib("xstring")
$handler.CopyLib("xmap")

$handler.dir = $FunctionalSTL
$handler.CopyLib("RE")
$handler.CopyLib("AC")
$handler.CopyLib("MC")

$handler.dir = $ModdedObjects
$handler.CopyLib("CC")


Write-Host "All Lib Folders Updated!!"
Write-Host "----------------------------------------------------------------------`n"
