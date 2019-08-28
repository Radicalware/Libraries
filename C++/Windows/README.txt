If you are on Unix, you must run

	snap install powershell --classic

If you want to easily use the CMake build script on Unix.

Bash is not object oriented and Python is not on windows by default.
I only wanted to make the build scripts once and it is much easier
to run the command above than to install Python on windows, so I 
will use PowerShell on Unix.



Version <Lib Count>.<New Addition Push Count>.<Hotfix>

Version = 9.X.X

eXtended_STL
	1. xvector  - eXtended std::vector
	2. xstring  - eXtended std::string
	3. xmap     - eXtended std::map
funcitonal_STL
	4. ac - array controller
	5. re - regex (string controller)
	6. mc - map controller
General_Purpose_Libs
	7. SYS - Key/Value Pair Argument Parsing
	8. OS  - Operating System handling
Mods
	9. cc  - modded termcolor by Ihor Kalnytskyi