
$module_path = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $module_path = "~/.local/share/powershell/Modules"
}else{
    $module_path = "$HOME\Documents\WindowsPowerShell\Modules"
}
Import-Module "$module_path\Arg_Struct.ps1" -Force
Import-Module "$module_path\PS_Builder.ps1" -Force

class Run_CMake
{

    $ArgStruct;
    
    Run_CMake($ArgStruct){
        $this.ArgStruct = $ArgStruct;
        $this.Copy_Files();

    }

    [void] Copy_Files(){ # For Myself and not to be of consern to the userbase
        if("$env:USERNAME`:$env:COMPUTERNAME" -eq "Scourge:WIT"){
            if($this.ArgStruct.Overwrite -ne $true){ # Do Not Overwrite Install Files (Preserve Them)
                   &"$PSScriptRoot\copy_projects.ps1"    
            }
        }
    }

    [Run_CMake] Print_Config(){

        Write-Host Options for: $this.ArgStruct.name;
        Write-Host ==================================;
        Write-Host "Debug Build = " $this.ArgStruct.debug;
        Write-Host "Clean Build = " $this.ArgStruct.clean;
        Write-Host ==================================;

        return $this;
    }

    [Run_CMake] Link_n_Compile(){
        $build = $(New-Object -TypeName PS_Builder -ArgumentList @($this.ArgStruct))

        if($this.ArgStruct.clean){
            $build.Configure_Build_Folder();
        }
        $build.Execute_CMake_Build_Config();
        $build.Compile_and_Link_Project();
        $build.Install_Files();
        $build.Return_Home();

        return $this;
    }

    [void] Execute(){

        if($this.ArgStruct.debug -eq $false){
            
            if($this.ArgStruct.is_unix){
                $exe_path = $PWD.ToString()+'/'+$this.ArgStruct.build_dir.ToString() + '/' +$this.ArgStruct.build_type.ToString() + '/' + $this.ArgStruct.name.ToString()
                Write-Host "$(/usr/bin/time -p $($exe_path) )";
            }else{
                $exe_path = $PWD.ToString()+'/'+$this.ArgStruct.build_dir.ToString() +'/'+$this.ArgStruct.build_type.ToString() + '/bin/'+ $this.ArgStruct.name.ToString()
                Write-Host "`nMilliseconds to Execute = " $(Measure-Command { &"$($exe_path).exe" | Write-Host }).Milliseconds;
            }
        }
    }
}






