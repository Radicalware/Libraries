#ifndef _OS_H_
#define _OS_H_

/*
* Copyright[2018][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/



// -------------------------------------------------------------------------------
// ALERT iterator will majorly slow down your performance if you don't
// optimize your compiler settings "-O2", else it will increase speed when
// not on windows (windows will give you even speeds with optimization else lag)
// also, on windows, be sure to remove the debugging for the iterator. 
// -------------------------------------------------------------------------------

#include "re.h" // From github.com/Radicalware
                // re.h has no non-std required libs
                // This is the only non-std lib required for OS.h


#include<iostream>
#include<vector>
#include<string>
#include<stdexcept>
#include<stdio.h>      // defines FILENAME_MAX or PATH_MAX

#include<fstream>      // file-handling

#include<dirent.h>     // read/write

#include<stdio.h>      // popen
#include<sys/stat.h>   // mkdir 
#include<unordered_map>

#include<assert.h>
#include <stdio.h>


#ifndef MSWINDOWS
    extern const std::string PLATFORM {"nix"};
    #include <unistd.h>
#else
    extern const std::string PLATFORM {"winbase"};
    #include <winbase.h>
#endif

using std::cout;
using std::endl;
using std::string;
using std::vector;


class OS
{
private:
    std::string m_command;
    std::string m_file_name;
    std::string m_file_data = "  ";

    char m_last_read = 'n';
    char m_write_method = 'a';

    int    m_argc;
    char** m_argv;
    
    std::unordered_map<std::string, std::vector<std::string> > m_args; // {base_arg, [sub_args]}

    std::vector<std::string> m_all_args; 
    std::string m_all_args_str;

    std::vector<std::string> m_bargs; // base args
    std::string m_bargs_str;
    std::vector< std::vector<std::string> > m_sub_args; // sub args
    std::string m_sub_args_str;

    std::string blank {""};

    std::vector<std::string> blank_vec;
    
    void dir_continued(string scan_start, vector<string>& vec_track, bool folders, bool files, bool recursive, bool star);
    
    void assert_folder_syntax(std::string folder1, std::string folder2 = "");

public:

    OS(int c_argc, char** c_argv);

    OS();

    ~OS();
    // ------------------------------------------
    OS open(std::string new_file_name, char write_method = 'a');
        // a = append     (append then writes like in python)
        // w = write mode (clears then writes like in python)

    std::string read(char content = 'n');

    std::string read_file();

    OS write(std::string content = "", char write_method = 'n');

    // ------------------------------------------

public:
    vector<string> dir(string folder_start, string mod1 = "n", string mod2 = "n", string mod3 = "n", string mod4 = "n");
        // dir(folder_to_start_search_from, &files_to_return,'r','f');
        // recursive = search foldres recursivly
        // folders   = return folders in search
        // files     = return files in search
        // star      = place a '*' in front of folders


    #ifndef MSWINDOWS
        std::string pwd();
    #else
        std::string pwd();
    #endif

    // Replace popen and pclose with _popen and _pclose for Windows.
    OS popen(const std::string command, char leave = 'p');
    // os.open(file_name).read()
    // os.popen(command).read()

    // ============================================================================================

    bool findFile(std::string file); // find based on ord:: syntax (no underscore)

    OS move_file(std::string old_location, std::string new_location = "" );
    OS copy_file(std::string old_location, std::string new_location = "" );
    OS clear_file(std::string content = "");
    OS delete_file(std::string content = "");

    OS mkdir(std::string folder);
    OS rmdir(std::string folder);
    // -------------------------------------------------------------------------------
    std::string file_data();
    std::string file_name();

    std::string command();
    std::string read_command();
    std::string cmd();
    // ============================================================================================
    // >>>> args

    OS set_args(int argc, char** argv);
    std::vector<std::string> argv();
    int argc();

    std::string operator[](int value); 
    std::unordered_map<std::string, std::vector<std::string> > args();
    
    // -------------------------------------------------------------------------------

    std::string argv_str();
    bool findArg(std::string find_arg);

    std::vector<std::string> keys();
    std::string keys_str();

    std::vector< std::vector<std::string> > keyValues();
    std::string keyValues_str();
    // -------------------------------------------------------------------------------
    
    std::string keyValue(std::string key, int i);

    std::vector<std::string> keyValues(std::string key);
    
    bool findKey(std::string key);
        // this is different than the ord::findKey
        // the ord::findKey will return false if the key has no value
        // os.findKey will return true as long as it exist so you can
        // have a header value that doesn't have sub-args
        // Also find will only return true if the key has data in the value
        // hence why I must iterate the base args vector

    bool findKeyValue(std::string key, std::string value);

    // -------------------------------------------------------------------------------
    template<class T = std::string>
    void p(T str);
    void d(int i = 0);
    // -------------------------------------------------------------------------------
};

extern OS os; // Header

#endif