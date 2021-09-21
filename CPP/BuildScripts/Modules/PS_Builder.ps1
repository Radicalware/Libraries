
$module_path = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $module_path = "~/.local/share/powershell/Modules"
}else{
    $module_path = "$HOME\Documents\WindowsPowerShell\Modules"
}
Import-Module "$module_path\Arg_Struct.ps1" -Force

$OFS = "`n"

Class PS_Builder
{
    $ArgStruct

    [string] $module_path;
    [string] $part_module_path;
    [string] $install_prefix;
    [string] $vcvars;
    [string] $cmake_command
    [string] $win_ver;
    [string] $comp_args;

    PS_Builder($ArgStruct){
        

        $this.ArgStruct = $ArgStruct;

        # Modify the paths below if they are not what you want.
        if($this.ArgStruct.is_unix){
            $this.cmake_command = "cmake"
            $this.win_ver = ''

            # $this.part_module_path  = 'opt/Radicalware/CMake_Modules' # no slash at start due to cmake issues  
            # $this.install_prefix = '/opt/Radicalware/Libraries'
            # $this.vcvars = ''
            # $this.module_path   = '/'+$this.part_module_path;
            # $this.comp_args = " -std=c++17 -Wfatal-errors -finput-charset=UTF-8 -fPIC -pthread"
        }else{ # Windows
            $this.cmake_command = "cmake.exe"
            $this.win_ver = '10.0.17763.0'
            $this.vcvars = 'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat'

            # $this.part_module_path  = 'Source/CMake/Modules'  
            # $this.install_prefix = 'C:/Source/CMake/Radicalware/Libraries'
            # $this.module_path   = 'C:/'+$this.part_module_path;
            # $this.comp_args = " /EHsc"
        }
        # build_dir = "Build \ <Windows/Nix> \ <Debug/Release>
    }

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Class Support (Would be Private Functions) 

    [string] Hide_Clock_Skew([string] $txt){
        # this is importent when you are working on remote files and
        # want to suppress errors that your timestamps are off
        return $($($txt -split "`n").ForEach({$($_.Tostring().Trim()| Select-String -NotMatch "Clock skew|modification time");}))
    }

    [void] Configure_Build_Folder(){
        if(Test-Path $this.ArgStruct.build_dir){
            Write-Host -ForegroundColor Green "[+] Deleting Up Old Project Folder:"$this.ArgStruct.build_type;
            Remove-Item $this.ArgStruct.build_dir -Recurse -Force -Confirm:$false ;
        }
        Write-Host -ForegroundColor Green "[+] Creating "$($this.build_type)" Folder";
        if($this.ArgStruct.is_unix){
            mkdir -p $([Regex]::Replace($this.ArgStruct.build_dir,"\\","/"))
        }else{
            mkdir $this.ArgStruct.build_dir | Out-Null;
        }
    }

    [void] Normalize_Encoding([string] $path_name){
        $(Get-ChildItem .\$path_name -Recurse -File).foreach({  Set-Content -Path $_.FullName -Encoding UTF8 -Value $(Get-Content $_.FullName) })
    }

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Class Usage (Would be Public Functions) 

    [void] Execute_CMake_Build_Config()
    {
        Set-Location $this.ArgStruct.build_dir

        if($this.ArgStruct.executable){
            $this.Normalize_Encoding("./Solution");
        }else{
            $this.Normalize_Encoding("./lib");
        }

        Set-Location $this.ArgStruct.build_dir;
        Write-Host -ForegroundColor Green "[+] Running CMake to configure a" $this.ArgStruct.build_type "Build";

        $this.cmake_command += " -DBUILD_TYPE=" + $this.ArgStruct.build_type.ToString();
        $this.cmake_command += " -DArgProjectName=" + $this.ArgStruct.name.ToString();
        if($this.ArgStruct.BuildAll){
            $this.cmake_command += " -DBUILD_ALL_PROJECTS=ON";
        }else{
            $this.cmake_command += " -DBUILD_ALL_PROJECTS=OFF";
        }
        
        $this.cmake_command += " ../../../";
        Write-Host $this.cmake_command;
        $ps_error_strings = @(
            "cmake.exe :"
            "At line:1 char:1"
            "\+ cmake.exe --config"
            "\+ ~~~~~~~~~~~~~~~"
            "    \+ CategoryInfo"
            "    \+ FullyQualified"
        )

        # Write-Host $cmake_command
        $cmake_out = "$(((Invoke-Expression "$($this.cmake_command) 2>&1" ) | Out-String).Split([Environment]::NewLine).Where({ $_ -ne `"`" -and !$([regex]::match($_, '^(('+$([string]::join(")|(",$ps_error_strings))+'))' )).Success }))".split("`n");
        
        Write-Host ([string]::Join("`n",$cmake_out));
        Set-Location $this.ArgStruct.base_dir
    }

    [void] Compile_and_Link_Project(){
        Set-Location $this.ArgStruct.build_dir

        if($this.ArgStruct.is_unix){ # I know, dumb that the -eq is required
            Write-Host -ForegroundColor Green "[+] Make is Building the Project " $this.ArgStruct.name
            $make_txt_out = $(make -j 2>&1)
            Write-Host $this.Hide_Clock_Skew($make_txt_out);
        }else{
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
            Write-Host -ForegroundColor Green "[+] CL.exe is the Building Project " $this.ArgStruct.name
            
            # Use this for fast small builds
            Write-Host "devenv $($this.ArgStruct.name + '.sln') /Build $($this.ArgStruct.build_type)"
            devenv $($this.ArgStruct.name + '.sln') /Build $this.ArgStruct.build_type | Write-Host
            
            # Use this for big large builds
            #BuildConsole.exe $($this.ArgStruct.name + '.sln') /cfg="$($this.ArgStruct.build_type)|x64" /NoLogo  | `
            #    Select-String -pattern "^\s|IncrediBuildAlwaysCreate|Temporary license|^\d+\>(Target|(\s+ (Deleting|Touching|Creating|All outputs are up-to-date|Building Custom Rule)))|^\d build system warnings|IncrediBuild|--------------------" -NotMatch | Write-Host
        }
        Set-Location $this.ArgStruct.base_dir
    }

    [void] Install_Files(){
        Set-Location $this.ArgStruct.build_dir
        Write-Host -ForegroundColor Green "[+] CMake is Installing Projct" $this.ArgStruct.name
        Write-Host "cmake -DBUILD_TYPE=$($this.ArgStruct.build_type) -P ./cmake_install.cmake"
        cmake "-DBUILD_TYPE=$($this.ArgStruct.build_type)" -P ./cmake_install.cmake | Write-Host
        Set-Location $this.ArgStruct.base_dir
    }

    [void] Return_Home(){
        Write-Host -ForegroundColor Green "[+] Returing Back to Home Folder"
        Set-Location $this.ArgStruct.base_dir
    }
};
