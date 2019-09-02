

class Arg_Struct
{
	[string] $name; 
	[bool]   $executable;
	[bool]   $shared_lib;
	[bool]   $debug;
	[bool]   $clean;
	[bool]   $Overwrite

	[string] $build_type;
	[string] $build_dir;
	[bool]   $is_unix;

	Arg_Struct([string] $proj_name, [bool[]]$bool_args){
		if($proj_name -eq ""){
			Write-Host "No project given";
			exit(0);
		}

		$this.name = $proj_name;

		$this.executable = $bool_args[0];
		$this.debug = $bool_args[1];
		$this.clean = $bool_args[2];
		$this.Overwrite = $bool_args[3];
		$this.shared_lib = $bool_args[4];

		$this.is_unix =  $($global:PSVersionTable.Platform -eq "Unix");

		$this.build_dir = "./Build";
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
