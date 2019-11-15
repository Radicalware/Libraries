#pragma once

// Lib: OS.h
// Version 1.4.1

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
// ALERT!! iterators will majorly slow down your performance if you don't
// optimize your compiler settings "-O2". If you are on Linux, it won't slow
// you donw but you won't get a speed boost either, so alway suse -O2!!
// -------------------------------------------------------------------------------

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#include<Windows.h>
#include<tchar.h> 
#include<stdio.h>
#include<strsafe.h>
#include<winnt.h>
#include<direct.h>
#include<stdlib.h>
#include<streambuf>
#else
#define NIX_BASE
#include<sys/types.h>
#include<unistd.h>
#include<pwd.h>
#include<dirent.h>     // read/write
#include<sys/ioctl.h>
#endif

#include<iostream>
#include<map>
#include<assert.h>
#include<errno.h>

#include<stdio.h>      // defines FILENAME_MAX or PATH_MAX
#include<fstream>      // file-handling
#include<cstdio>       // rename
#include<sys/stat.h>   // mkdir 
#include<stdio.h>      // popen

#include "xvector.h"
#include "xstring.h"

#include "handlers/CMD.h"
#include "handlers/File.h"
#include "dir_support/Dir_Type.h"


// Note: All Static Functions / Objects start with a CAP
class OS : public OS_O::Dir_Type
{
private:
    char m_last_read;

    static void Dir_Continued(const xstring scan_start, xvector<xstring>& vec_track, \
        const bool folders, const bool files, const bool recursive);

public:
    OS_O::File file; // .name()   .data()     
    OS_O::CMD  cmd;  // .cmd()    .in()     .out().   .err()    .err_message()

    OS();
    ~OS();

    static xvector<int> Console_Size(); // columns, rows

    static bool File_Syntax(const xstring& file);
    static bool File_List_Syntax(const xvector<xstring>& files);
    static xstring Full_Path(const xstring& file);

    // ---------------------------------------------------------------------------------------------
    // Bash Style OS Commands

    void touch(const xstring& new_file = "");
    static void MKDIR(const xstring& folder = "");

    static void CP(const xstring& old_location, const xstring& new_location);
    static void MV(const xstring& old_location, const xstring& new_location);
    static void RM(const xstring& del_file);
    
    // ---------------------------------------------------------------------------------------------
    // Open & Read/Write Files

    // r = read,     w = write,     a = append,     d = default (read)
    OS open(const xstring& new_file_name, const char write_method = 'd');
    OS close();

    xstring read(const char content = 'd', bool close_file = false);
private:
    xstring inst_read();

public:
    static xstring Fast_Read(const xstring& file_name, bool re_try = false); // Good for reading files text only
    static xstring Stat_Read(const xstring& file_name);   // Good for text and binaries on Linux
    static xstring Stream_Read(const xstring& file_name); // Good for text and binaries on Linux and Windows

    OS write(const xstring& content = "", bool store = false);

    // ---------------------------------------------------------------------------------------------
    // Dir Parsing

    static xvector<xstring> Dir(const xstring& folder_start, \
        const char mod1 = 'n', const char mod2 = 'n', const char mod3 = 'n');
    // dir(folder_to_start_search_from, mod can match for any of the following 3);
    // r = recursive = search dirs recursivly
    // f = files     = return files in search
    // d = directory = return dirs in search

    // ---------------------------------------------------------------------------------------------
    // Console Command and Return
    OS& popen(const xstring& command, char leave = 'd'); 
    xstring operator()(const xstring& command, const char leave = 'd');

    // ============================================================================================
    // Below are methods for short-hand printing;
    template<typename T>
    static void P(const T& input);

    template <typename First, typename... Rest>
    static void P(const First& input1, const Rest&&... input2);

    // ============================================================================================
    // Filesystem Managment (use "Bash style OS commands" above for shorthand)

    static void Assert_Folder_Syntax(const xstring& folder1);

    static void Move_File(const xstring& old_location, const xstring& new_location);
    static void Move_Dir(const xstring& old_location, const xstring& new_location);

    static void Copy_File(const xstring& old_location, const xstring& new_location);
    static void Copy_Dir(const xstring& old_location, const xstring& new_location);

    static void Remove_File(const xstring& content = "");
    static void Remove_Dir(const xstring& folder = "");

    static void Clear_File(const xstring& content = "");
    // ============================================================================================
};

// Radicalware Product
