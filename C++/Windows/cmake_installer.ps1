param (
    [string]$project,
    [string]$clean_build = "true", # can't use switch type and specify false explicity
    [string]$build_type = "Release"
)

# This is just something I use, not inteneded for users for installing my libs.
if("$env:USERNAME`:$env:COMPUTERNAME" -eq "Scourge:WIT"){

    $update_cmake_build = Read-Host 'Preserve Install Files & Delete CMake Builds (Default = yes)'

    if($update_cmake_build -eq '' -or ($update_cmake_build | Select-String -Pattern "^(-?)[yY]((es)?)$") -eq $true){
       C:\source\include\copy_libs.ps1
    }
}

if($project -eq ""){
    Write-Host "No project given";
    exit(0);
}

# Modify the paths below if they are not what you want.
$find_packages  = 'source/CMake/FindPackages'                  # find_path # C drive is added in the CMake files
$install_prefix = 'C:/source/CMake/Radicalware/Libraries/cpp'  # install_prefix

$vcvars_path = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat'

Class Build
{
    [string] $name;
    [string] $build_type;
    [string] $find_path;
    [string] $install_prefix;
    [string] $vcvars;
    [string] $build_dir;

    Build([string[]] $cmake_init, [string] $vcvars){

        $this.build_type        = $cmake_init[0];
        $this.name              = $cmake_init[1];
        $this.find_path         = $cmake_init[2];
        $this.install_prefix    = $cmake_init[3];
        $this.vcvars            = $vcvars;
		$this.build_dir         = ".\Build\$($this.build_type)"
    }

    [void] Configure_Build_Folder(){
        if(Test-Path $this.build_dir){
            Write-Host -ForegroundColor Green "[+] Deleting Up Old Project Folder:"$this.build_type;
            Remove-Item $this.build_dir -Recurse -Force -Confirm:$false ;
        }
        Write-Host -ForegroundColor Green "[+] Creating "$($this.build_type)" Folder";
        mkdir $this.build_dir | Out-Null;
    }

    [void] Execute_CMake_Build_Config(){
        Set-Location $this.build_dir
        Write-Host -ForegroundColor Green "[+] Running CMake to configure a" $this.build_type "Build"
        $(cmake.exe --config $this.build_type `
            -DBUILD_TYPE="$($this.build_type)" `
            -DFIND_PATH="$($this.find_path)" `
            -DCMAKE_INSTALL_PREFIX="$($this.install_prefix)" `
            -DINSTALL_PREFIX="$($this.install_prefix)" `
            -DINCLUDE_PATH="$($this.install_prefix)/include" `
            -DLIB_BIN_PATH="$($this.install_prefix)/bin/$($this.build_type)" `
            -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE `
            ..\..\)| Write-Host
            # -DOPERATIONAL_DIR="$($(Get-Location).Path)" `
            # -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE `
            
    }

    [void] Compile_and_Link_Project(){
        if($env:__DOTNET_ADD_64BIT -ne 1){
            Write-Host -ForegroundColor Green "[+] Configuring CL.exe for x86_64 Compiling"
            cmd.exe /c "echo >  %temp%\vcvars.txt"
            cmd.exe /c "call `"$($this.vcvars)`" && set > %temp%\vcvars.txt"
        
            Get-Content "$env:temp\vcvars.txt" | Foreach-Object {
                if ($_ -match "^(.*?)=(.*)$") {
                    Set-Content "env:\$($matches[1])" $matches[2]
                }
            }
        }else{
            Write-Host -ForegroundColor Green "[+] CL.exe Is Already Configured for x86_64 Compiling"
        }

        Write-Host -ForegroundColor Green "[+] CL.exe is the Building Project"
        devenv $($this.name + '.sln') /build $this.build_type | Write-Host
    }

    [void] Install_Build_Files(){
        Write-Host -ForegroundColor Green "[+] CMake is Installing Projct"
        cmake -P .\cmake_install.cmake | Write-Host
    }

    [void] Return_Home(){
        Write-Host -ForegroundColor Green "[+] Returing Back to Home Folder"
        Set-Location ..\..\      
    }
};

$build = [Build]::new(
    @(
        $build_type,    # release/debug (release is default)
        $project,       # I use ${THIS} instead of ${PROJECT_NAME}
        $find_packages  # find_path
        $install_prefix # install_prefix
    ),
    $vcvars_path  # Path to vcvars.exe (See top of the file)
);

if($clean_build -eq "true"){
    $build.Configure_Build_Folder();
}
$build.Execute_CMake_Build_Config();
$build.Compile_and_Link_Project();
$build.Install_Build_Files();
$build.Return_Home();

	