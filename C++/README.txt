If you are on Unix, you must run

	snap install powershell --classic

If you want to easily use the CMake build script on Unix.

Bash is not object oriented and Python is not on windows by default.
I only wanted to make the build scripts once and it is much easier
to run the command above than to install Python on windows, so I 
will use PowerShell on Unix.


Version <Lib Count>.<New Addition Push Count>.<Hotfix>

Version = 12.X.X

eXtended_STL
	 1. xvector   - eXtended std::vector
	 2. xstring   - eXtended std::string
	 3. xmap      - eXtended std::map
funcitonal_STL
	 4. ac        - array controller
	 5. re        - regex (string controller)
	 6. mc        - map controller
General_Purpose_Libs
	 7. SYS       - Key/Value Pair Argument Parsing
	 8. OS        - Operating System handling
	 9. Nexus     - Thread Pool with Indexing & Exception Handling
	10. Timer     - Used for tracking the duration of a processes
Mods
	11. cc        - A modified version of termcolor by Ihor Kalnytskyi
Modules	
	12. QtImage   - A module that handles images on Qt Objects


--------------------------------------------------------------------------------------------
On Nix, add the following to your bash env (bashrc/bash_profile/etc)
--------------------------------------------------------------------------------------------
# Library Path for Source Code
export LIBRARY_PATH="/usr/local/include":$LIBRARY_PATH
export LIBRARY_PATH="/opt/Radicalware/Libraries/cpp/include":$LIBRARY_PATH
export LIBRARY_PATH="/opt/Radicalware/Libraries/cpp/code/Projects":$LIBRARY_PATH

export CPATH="/opt/Radicalware/Libraries/cpp/include":$CPATH
export CPATH="/opt/Radicalware/Libraries/cpp/code/Projects":$CPATH

export LD_LIBRARY_PATH="/opt/Radicalware/Tools/Release/bin":$LD_LIBRARY_PATH
export LD_RUN_PATH="/opt/Radicalware/Tools/Release/bin":$LD_RUN_PATH