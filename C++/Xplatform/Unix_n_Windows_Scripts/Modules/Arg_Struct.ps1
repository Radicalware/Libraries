

class Arg_Struct
{
    # --------- user defined ---------
    [string] $name; 
    [bool]   $executable;
    [bool]   $debug;
    [bool]   $clean;
    [bool]   $Overwrite

    [bool]   $noCmake
    [bool]   $noMake
    [bool]   $noInstall
    # ---------------------------------
    [string] $build_type;
    [bool]   $is_unix;

    [string] $base_dir;
    [string] $build_dir;
    # ---------------------------------

    Arg_Struct([string] $proj_name, [string] $base_dir, [bool[]]$bool_args)
    {
        $this.base_dir = $base_dir;

        if($proj_name -eq ""){
            Write-Host "No project given";
            exit(0);
        }

        $this.name = $proj_name;
        # exe, debug, clean, overwrite, nCmake, noMake, noInstall

        $this.executable =  $bool_args[0];
        $this.debug =       $bool_args[1];
        $this.clean =       $bool_args[2];
        $this.Overwrite =   $bool_args[3];

        $this.noCmake =     $bool_args[4];
        $this.noMake =      $bool_args[5];
        $this.noInstall =   $bool_args[6];

        $this.is_unix =  $($global:PSVersionTable.Platform -eq "Unix");

        $this.build_dir  = $this.base_dir
        $this.build_dir += "/Build";
        if($this.is_unix){
            $this.build_dir += "/Nix"
        }else{
            $this.build_dir += "\Windows"
        }
        if($this.debug){
            $this.build_type = "Debug"
        }else{
            $this.build_type = "Release"
        }


        $this.build_dir += '/'+$this.build_type
        if(!(Test-Path $this.build_dir)){
            New-Item $this.build_dir -ItemType Directory | Out-Null
        }
    }
}
