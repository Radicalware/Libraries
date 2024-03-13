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
     4. Memory      - eXtended versions of Memory Objects
FuncitonalSTL
     5. ac          - array controller
     6. re          - regex (string controller)
     7. mc          - map controller
GPU
     8. ImportCUDA  - Gives control of Host/Device mem allocation
     9. CudaBridge  - Object Control
    10. Stats       - CPU/GPU data processing Object
GeneralPurposeLibs
    11. Macros      - Helps in handling pointers and Exception Handling
    12. SYS         - Key/Value Pair Argument Parsing
    13. OS          - Operating System handling
    14. Nexus       - Thread Pool with Indexing & Exception Handling
    15. Socket      - This makes client/server protocols very easy to use
    16. Timer       - Used for tracking the duration of a processes
    17. Date        - Converts dates to ints and vice verca
    18. JSON        - Moves data between API JSON and DB BSON
    19. Stash       - Easy tool to manage your Mongo DB
    20. AES         - Manage AES Cipher Text for Password Managment
    21. Mirror      - Forwards & Backwards Indexing container
ModdedObjects
    22. cc          - A modified version of termcolor by Ihor Kalnytskyi
    23. re2         - Modded version deleted in favor of vcpkg's version
Modules
    Qt
        24. QtImage - Formats the image to fill and perserve aspect ratio

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

