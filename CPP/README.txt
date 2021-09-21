If you are on Unix, you must run

    snap install PowerShell --classic

If you want to easily use the CMake build script on Unix.

Bash is not object oriented and Python is not on windows by default.
I only wanted to make the build scripts once and it is much easier
to run the command above than to install Python on windows, so I 
will use PowerShell on Unix.

Also, to use the PowerShell scripts, you must run "prep_modules.ps1"
which will add the modules to your PowerShell $PATH

Version <Lib Count>.<New Addition Push Count>.<Hotfix>

Version = <Lib Count>.<Enhancement>.<Hotfix>


Extended_STL
     1. xvector     - eXtended std::vector
     2. xstring     - eXtended std::string
     3. xmap        - eXtended std::map
FuncitonalSTL
     4. ac          - array controller
     5. re          - regex (string controller)
     6. mc          - map controller
GeneralPurposeLibs
     7. Macros      - Helps in handling pointers and Exception Handling
     8. SYS         - Key/Value Pair Argument Parsing
     9. OS          - Operating System handling
    10. Nexus       - Thread Pool with Indexing & Exception Handling
    11. Socket      - This makes client/server protocols very easy to use
    12. Timer       - Used for tracking the duration of a processes
    13. Date        - Converts dates to ints and vice verca
    14. CudaBridge  - Handles memory between host and device
    15. JSON        - Moves data between API JSON and DB BSON
    16. Stash       - Easy tool to manage your Mongo DB
ModdedObjects
    17. cc          - A modified version of termcolor by Ihor Kalnytskyi
    18. re2         - Modded version deleted in favor of vcpkg's version
Modules
    Qt
        19. QtImage - Formats the image to fill and perserve aspect ratio

--------------------------------------------------------------------------------------------
On Nix, add the following to your bash env (bashrc/bash_profile/etc)
--------------------------------------------------------------------------------------------

export PATH="/opt/Radicalware/Applications/Build/Release":$PATH

export LIBRARY_PATH="/usr/local/include":$LIBRARY_PATH
export LIBRARY_PATH="/opt/Radicalware/Libraries/Include":$LIBRARY_PATH
export LIBRARY_PATH="/opt/Radicalware/Libraries/Projects":$LIBRARY_PATH

export CPATH="/opt/Radicalware/Libraries/Include":$CPATH
export CPATH="/opt/Radicalware/Libraries/Projects":$CPATH

export LD_LIBRARY_PATH="/opt/Radicalware/Libraries/Build/Release/bin":$LD_LIBRARY_PATH
export LD_RUN_PATH="/opt/Radicalware/Libraries/Build/Release/bin":$LD_RUN_PATH

