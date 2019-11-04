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
#else
#define NIX_BASE
#include<sys/stat.h>
#include<sys/types.h>
#include<unistd.h>
#include<pwd.h>
#include<unistd.h>
#include<dirent.h>     // read/write
#include<sys/ioctl.h>
#endif

#include<iostream>
#include<assert.h>

#include<stdio.h>      // defines FILENAME_MAX or PATH_MAX
#include<fstream>      // file-handling
#include<cstdio>       // rename
#include<sys/stat.h>   // mkdir 
#include<stdio.h>      // popen

#include "xstring.h"
#include "xvector.h"

#include "dir_support/Dir_Type.h"
#include "dir_support/File_Names.h"

class OS : public Dir_Type
{
private:
    xstring m_command;
    xstring m_std_out;
    xstring m_std_err;
    xstring m_err_message;

    xstring m_file_name;
    xstring m_file_data;

    char m_last_read;
    char m_write_method;

    bool m_rexit;

    static bool instance_started;
    static OS* inst;

    void dir_continued(const xstring scan_start, xvector<xstring>& vec_track, \
        const bool folders, const bool files, const bool recursive);

public:
    OS();

    ~OS();

    xvector<int> console_size(); // columns, rows
    void set_file_regex(bool rexit); // asserting file-regex consumes a lot of time
                                     // only turn on when you are parsing user input
    bool file_regex_status();
    bool file_syntax(const xstring& file);
    bool file_list_syntax(const xvector<xstring>& files);

    xstring full_path(const xstring& file);

    // ---------------------------------------------------------------------------------------------
    // Bash Style OS Commands

    void touch(const xstring& new_file = "");
    void mkdir(const xstring& folder = "");

    void cp(const xstring& old_location, const xstring& new_location);
    void mv(const xstring& old_location, const xstring& new_location);
    void rm(const xstring& del_file);
    
    // ---------------------------------------------------------------------------------------------
    // Open & Read/Write Files

    OS open(const xstring& new_file_name, const char write_method = 'a');
    // a = append     (append then writes like in python)
    // w = write mode (clears then writes like in python)

    xstring read(const char content = 'n');
    xstring read_file();
    OS write(const xstring& content = "", const char write_method = 'n');

    // ---------------------------------------------------------------------------------------------
    // Dir Parsing

    xvector<xstring> dir(const xstring& folder_start, \
        const char mod1 = 'n', const char mod2 = 'n', const char mod3 = 'n');
    // dir(folder_to_start_search_from, mod can match for any of the following 3);
    // r = recursive = search dirs recursivly
    // f = files     = return files in search
    // d = directory = return dirs in search

    // ---------------------------------------------------------------------------------------------
    // Console Command and Return
    OS popen(const xstring& command, char leave = 'p');
    xstring operator()(const xstring& command);
    template<typename T>
    void p(const T& input);
    template<typename T, typename X>
    void p(const T& input1, const X& input2);
    // shorthand for os.popen(command).read()

    // ============================================================================================
    // Filesystem Managment (use "Bash style OS commands" above for shorthand)

    void move_file(const xstring& old_location, const xstring& new_location);
    void move_dir(const xstring& old_location, const xstring& new_location);

    void copy_file(const xstring& old_location, const xstring& new_location);
    void copy_dir(const xstring& old_location, const xstring& new_location);

    void delete_file(const xstring& content = "");
    void delete_dir(const xstring& folder = "");

    void clear_file(const xstring& content = "");

    // -------------------------------------------------------------------------------
    // Getters

    xstring file_data();
    xstring file_name();

    xstring cli_command();

    xstring std_in();
    xstring std_out();
    xstring std_err();

    xstring err_message();
    // ============================================================================================
};

// Radicalware Product
