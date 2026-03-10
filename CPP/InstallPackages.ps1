param(
    [Parameter()]
    [ValidateSet('Install', 'Remove', 'Update')]
    [string]$Action = 'Install'
)

set64
class Core
{
    static[string[]] $SvPackages = @(
        "boost",
        "zlib",
        "libbson",
        "openssl",
        "boost-system",
        "boost-date-time",
        "boost-regex",
        "boost-interprocess",
        "atlmfc",
        "vtk[atlmfc]",
        "mongo-cxx-driver",
        "vulkan",
        "nlohmann-json",
        "cpprestsdk",
        "glm",
        "abseil"
    );

    static[void] Install()
    {
        foreach($LsPackage in [Core]::SvPackages){
            vcpkg install --triplet x64-windows $LsPackage | Out-Host
        }
    }

    static[void] Remove()
    {
        $LsConfirmation = (Read-Host "Proceed with removal of all packages('yes')?").Trim().ToLower()
        if ($LsConfirmation -ne 'y' -and $LsConfirmation -ne 'yes') {
            Write-Host "Operation cancelled. (needed a 'y' or 'yes')"
            exit 1
        }
        
        foreach($LsPackage in [Core]::SvPackages){
            if($(vcpkg list $LsPackage).Length -gt 0){
                vcpkg remove $LsPackage --recurse | Out-Host
            }
        }
    }

    static[void] CleanInstall()
    {
        [Core]::Remove()
        [Core]::Install()
    }
};

switch ($Action) {
    'Install' {
        [Core]::Install()
    }
    'Remove' {
        [Core]::Remove()
    }
    'Update' {
        [Core]::CleanInstall()
    }
}

