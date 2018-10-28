#pragma once

// Lib: OS.h
// Version 1.2.1

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

#include "re.h" // From github.com/Radicalware
				// re.h has no non-std required libs
				// This is the only non-std lib required for OS.h

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#include <windows.h>
#include <msclr/marshal.h>
//using namespace System;
//using namespace System::Configuration;

#else
#define NIX_BASE
#include <unistd.h>
#include<dirent.h>     // read/write
#endif


#include<iostream>
#include<vector>
#include<assert.h>

#if defined(NIX_BASE)
	#include "./support_os/File_Names.h"
#elif defined(WIN_BASE)
	#include ".\support_os\File_Names.h"
#endif



class OS
{
private:
	std::string m_command;
	std::string m_file_name;
	std::string m_file_data;

	char m_last_read = 'n';
	char m_write_method = 'a';

	std::wstring m_wstr;

	enum dir_type
	{
		dir_none,
		dir_file,
		dir_folder
	};


	File_Names id_files(std::string old_location, std::string new_location = "");

	void dir_continued(const std::string scan_start, std::vector<std::string>& vec_track, \
		const bool folders, const bool files, const bool recursive);

public:

	OS();
	~OS();

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

	std::string pwd();
	std::vector<std::string> dir(const std::string folder_start, const std::string& mod1 = "n", \
		const std::string& mod2 = "n", const std::string& mod3 = "n");
	// dir(folder_to_start_search_from, mod can match for any of the following 3;
	// recursive = search foldres recursivly
	// folders   = return folders in search
	// files     = return files in search

	// ---------------------------------------------------------------------------------------------
	// Console Command and Return

	OS popen(const std::string& command, char leave = 'p');
	std::string operator()(const std::string& command);
	// shorthand for os.popen(command).read()

	// ============================================================================================
	// Bools for identifying data

	dir_type has(const std::string& file);
	bool has_file(const std::string& file);
	bool has_folder(const std::string& folder);

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

	std::string command();
	std::string read_command();
	std::string cmd();
	// ============================================================================================
};

extern OS os;
