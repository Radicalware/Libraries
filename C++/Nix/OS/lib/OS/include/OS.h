#pragma once

// Lib: OS.h
// Version 1.4.0

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

// re.h is from github.com/Radicalware
// This is the only non-std lib required for OS.h

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#include <windows.h>
#include <msclr/marshal.h>
#else
#include<unistd.h>
#include<dirent.h>     // read/write
#endif


#include<iostream>
#include<vector>
#include<assert.h>

#include "re.h"
#include "support_os/Dir_Type.h"
#include "support_os/File_Names.h"


class OS : public Dir_Type
{
private:
    std::string m_command;
    std::string m_std_out;
    std::string m_std_err;
    std::string m_err_message;

    std::string m_file_name;
    std::string m_file_data;

    char m_last_read = 'n';
    char m_write_method = 'a';
    bool m_rexit = false;

    std::wstring m_wstr;

    File_Names id_files(std::string old_location, std::string new_location = "");

    void dir_continued(const std::string scan_start, std::vector<std::string>& vec_track, \
        const bool folders, const bool files, const bool recursive);

    void coppier(const std::vector<std::string>& folders, const std::vector<std::string>& files, 
        const std::vector<std::string>& dir_items, File_Names& fls);

public:

    OS();
    ~OS();

    void set_file_regex(bool rexit); // asserting file-regex consumes a lot of time
                                     // only turn on when you are parsing user input
    bool file_regex_status();
    bool file_syntax(const std::string& file);
    bool file_list_syntax(const std::vector<std::string>& files);

    // ---------------------------------------------------------------------------------------------
    // Bash Style OS Commands

    OS touch(const std::string& new_file = "");
    OS mkdir(const std::string& folder = "");

    OS cp(const std::string& old_location, const std::string& new_location = "");
    OS mv(const std::string& old_location, const std::string& new_location = "");
    OS rm(const std::string& new_file);

    // ---------------------------------------------------------------------------------------------
    // Open & Read/Write Files

    OS open(const std::string& new_file_name, const char write_method = 'a');
    // a = append     (append then writes like in python)
    // w = write mode (clears then writes like in python)

    std::string read(const char content = 'n');
    std::string read_file();
    OS write(const std::string& content = "", const char write_method = 'n');

    // ---------------------------------------------------------------------------------------------
    // Dir Parsing

    std::string bpwd(); // binary pwd
    std::string pwd();  // user pwd
    std::string home(); // home dir

    std::vector<std::string> dir(const std::string folder_start, \
        const char mod1 = 'n', const char mod2 = 'n', const char mod3 = 'n');
    // dir(folder_to_start_search_from, mod can match for any of the following 3);
    // r = recursive = search dirs recursivly
    // f = files     = return files in search
    // d = directory = return dirs in search

    // ---------------------------------------------------------------------------------------------
    // Console Command and Return
    OS popen(const std::string& command, char leave = 'p');
    std::string operator()(const std::string& command);
    template<typename T>
    void p(const T& input);
    template<typename T, typename X>
    void p(const T& input1, const X& input2);
    // shorthand for os.popen(command).read()

    // ============================================================================================
    // Filesystem Managment (use "Bash style OS commands" above for shorthand)

    OS move_file(const std::string& old_location, const std::string& new_location);
    OS move_dir(const std::string& old_location, const std::string& new_location);

    OS copy_file(const std::string& old_location, const std::string& new_location);
    OS copy_dir(const std::string& old_location, const std::string& new_location);

    OS delete_file(const std::string& content = "");
    OS delete_dir(const std::string& folder = "");

    OS clear_file(const std::string& content = "");

    // -------------------------------------------------------------------------------
    // Getters

    std::string file_data();
    std::string file_name();

    std::string cli_command();
    std::string std_in();
    std::string std_out();
    std::string std_err();
    std::string err_message();
    // ============================================================================================
};

extern OS os;


