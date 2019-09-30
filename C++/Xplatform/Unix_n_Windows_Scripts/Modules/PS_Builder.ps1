
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
	[string] $vcpkg_path;
    [string] $vcvars;
    [string] $cmake_command

    PS_Builder($ArgStruct){
		

		$this.ArgStruct = $ArgStruct;

		# Modify the paths below if they are not what you want.
		if($this.ArgStruct.is_unix){
			$this.part_module_path  = 'usr/share/cmake/Modules'  
			$this.install_prefix = '/opt/Radicalware/Libraries/cpp'
			$this.vcpkg_path = "$global:HOME/lp/vcpkg/scripts/buildsystems/vcpkg.cmake"
			$this.vcvars = ''
            $this.module_path   = '/'+$this.part_module_path;
            $this.cmake_command = "cmake"

		}else{
			$this.part_module_path  = 'source/CMake/Modules'  
			$this.install_prefix = 'C:/source/CMake/Radicalware/Libraries/cpp'
			$this.vcpkg_path = "C:/source/lp/vcpkg/scripts/buildsystems/vcpkg.cmake"
			$this.vcvars = 'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat'
            $this.module_path   = 'C:/'+$this.part_module_path;
            $this.cmake_command = "cmake.exe"

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

    [void] Execute_CMake_Build_Config(){

        if($this.ArgStruct.executable){
            $this.Normalize_Encoding("./Solution");
        }else{
            $this.Normalize_Encoding("./lib");
        }

        Set-Location $this.ArgStruct.build_dir
        Write-Host -ForegroundColor Green "[+] Running CMake to configure a" $this.ArgStruct.build_type "Build"
        
        $this.cmake_command += " --config -Wno-unused-variable " + $this.ArgStruct.build_type `
            + " -DBUILD_TYPE=" + $this.ArgStruct.build_type `
            + " -DCMAKE_INSTALL_PREFIX=" + $this.install_prefix `
            + " -DINSTALL_PREFIX=" + $this.install_prefix `
            + " -DEXT_INCLUDE_PATH=" + $this.install_prefix + "/include" `
            + " -DEXT_BIN_PATH=" + $this.install_prefix + "/bin/$($this.build_type)" `
            + " -DMODULE_PATH=" + $this.module_path 
			# + " -DCMAKE_TOOLCHAIN_FILE=" + $this.vcpkg_path


		if(!$this.ArgStruct.is_unix){
			$this.cmake_command += " -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE"
		}
		if($this.ArgStruct.shared_lib -or $this.ArgStruct.executable){
			$this.cmake_command += " -DBUILD_SHARED_LIBS=TRUE";
		}
		if(!$this.ArgStruct.executable){
            $this.cmake_command += " -DPART_MODULE_PATH=" + $this.part_module_path `
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
    }

    [void] Compile_and_Link_Project(){
        if($this.ArgStruct.is_unix){ # I know, dumb that the -eq is required
            Write-Host -ForegroundColor Green "[+] Make is Building the Project" 
            $make_txt_out = $(make 2>&1)
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
            Write-Host -ForegroundColor Green "[+] CL.exe is the Building Project"
            devenv $($this.ArgStruct.name + '.sln') /build $this.ArgStruct.build_type | Write-Host
        }
    }

    [void] Install_Files(){
        Write-Host -ForegroundColor Green "[+] CMake is Installing Projct" $this.ArgStruct.name
        cmake -P ./cmake_install.cmake | Write-Host
    }

    [void] Return_Home(){
        Write-Host -ForegroundColor Green "[+] Returing Back to Home Folder"
        Set-Location ../../../
    }
};
