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


#include "OS.h"
#include "re.h" // From github.com/Radicalware
				// re.h has no non-std required libs
				// This is the only non-std lib required for os.h

#include<iostream>
#include<vector>
#include<string>
#include<stdexcept>
#include<stdio.h>      // defines FILENAME_MAX or PATH_MAX
#include<string>

#include<fstream>      // file-handling

#if defined(WIN_BASE)
#include<Windows.h>
#include<msclr/marshal.h>
#include<tchar.h> 
#include<stdio.h>
#include<strsafe.h>
#include<winnt.h>
#endif


#include<stdio.h>      // popen
#include<sys/stat.h>   // mkdir 
#include<unordered_map>

#include<assert.h>
#include<stdio.h>


OS::OS() {};
OS::~OS() {};

// ------------------------------------------

OS OS::open(const std::string& new_file_name, const char write_method) {
	// a = append     (append then writes like in python)
	// w = write mode (clears then writes like in python)

	m_write_method = write_method;
	m_file_name = new_file_name;
	switch (m_write_method) {
	case 'a':  m_write_method = 'a';
		break;
	case 'w':  m_write_method = 'w';
		this->clear_file(m_file_name);
	}
	m_last_read = 'f';
	return *this;
}

// os.open(file_name).read()
// os.popen(command).read()
std::string OS::read(char content) {
	content = (content == 'n') ? m_last_read : content;

	switch (content) {
	case 'f':  return this->read_file(); break;
	case 'c':  return m_command; break;
	default:  return "none";
	}
}

std::string OS::read_file() {
	std::ifstream os_file(m_file_name);
	std::string line;
	m_file_data.clear();

	if (os_file.is_open()) {
		while (getline(os_file, line)) {
			m_file_data += line + '\n';
		}
		os_file.close();
	}

	return m_file_data;
}

OS OS::write(const std::string& content, char write_method) {

	m_write_method = (write_method == 'n') ? m_write_method : write_method;

	switch (m_write_method) {
	case 'a':  m_write_method = 'a';
		m_file_data = this->read() + content;
		break;

	case 'w':  m_write_method = 'w';
		m_file_data = content;
		this->clear_file(m_file_name);
	}

	const char* content_ptr = &m_file_data[0];

	std::ofstream os_file(m_file_name);
	if (os_file.is_open()) {
		os_file << content_ptr;
		os_file.close();
	}
	m_last_read = 'f';
	return *this;
}


using std::cout;
using std::endl;
// -------------------------------------------------------------------------------------------
#if defined(NIX_BASE)
void OS::dir_continued(const std::string& scan_start, std::vector<std::string>& track_vec, \
	const bool folders, const bool files, const bool recursive) {

	DIR *current_dir = opendir(scan_start.c_str());
	// starting dir given as a std::string
	// "dir_item" can be anything in the "current_dir" 
	// such as a new folder, file, binary, etc.
	while (struct dirent *dir_item_ptr = readdir(current_dir)) {
		std::string dir_item = (dir_item_ptr->d_name);
		// structure points to the getter to retrive the dir_item's name.
		if (dir_item != "." and dir_item != "./" and dir_item != "..") {
			if (dir_item_ptr->d_type == DT_DIR) {
				if (folders) {
					track_vec.push_back(scan_start + "/" + dir_item + "/");
				}
				if (recursive) {
					this->dir_continued(scan_start + "/" + dir_item, \
						track_vec, folders, files, recursive);
					// recursive function
				}
			}
			else if (dir_item == "read") {
				break; // full dir already read, leave the loop
			}
			else if (files) {
				track_vec.push_back(scan_start + "/" + dir_item);
			}
		}
	}
	closedir(current_dir);
}
#elif defined(WIN_BASE)
void OS::dir_continued(const std::string& folder_start, std::vector<std::string>& track_vec, \
	const bool folders, const bool files, const bool recursive) {

	WIN32_FIND_DATA find_data;
	LARGE_INTEGER filesize;
	TCHAR dir_size_tchar[MAX_PATH];
	HANDLE hFind = INVALID_HANDLE_VALUE;

	// If the directory is not specified as a command-line argument,
	// print usage.

	std::string item = folder_start;
	std::wstring wstr(item.begin(), item.end());
	assert(item.size() < MAX_PATH - 3);

	// Prepare string for use with FindFile functions.  First, copy the
	// string to a buffer, then append '\*' to the directory name.
	StringCchCopy(dir_size_tchar, MAX_PATH, wstr.c_str());
	StringCchCat(dir_size_tchar, MAX_PATH, TEXT("\\*"));

	// Find the first file in the directory.
	hFind = FindFirstFile(dir_size_tchar, &find_data);

	auto t_path_to_str_path = [&find_data]() -> std::string {
		std::string path_str;
		int count = 0;
		while (find_data.cFileName[count] != '\0') {
			path_str += find_data.cFileName[count];
			++count;
		}
		return path_str;
	};

	// check to make sure it is there
	if (INVALID_HANDLE_VALUE != hFind) {
		// List all the files in the directory with some info about them.
		std::string file_path_str;
		do
		{
			if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
				file_path_str = t_path_to_str_path() + "\\";
				if (file_path_str == ".\\" || file_path_str == "..\\")
					continue;
				std::string full_path = folder_start + "\\" + file_path_str;
				if (folders)
					track_vec.push_back(full_path);
				if (recursive)
					dir_continued(full_path, track_vec, folders, files, recursive);
			}
			else {
				if (files)
					track_vec.push_back(folder_start + "\\" + t_path_to_str_path());
			}
		} while (FindNextFile(hFind, &find_data) != 0);

		FindClose(hFind);
	}
};
#endif

std::vector<std::string> OS::dir(const std::string& folder_start, const std::string& mod1,
	const std::string& mod2, const std::string& mod3) {
	// dir(folder_to_start_search_from, &files_to_return,'r','f');
	// recursive = search foldres recursivly
	// folders   = return folders in search
	// files     = return files in search

	this->assert_folder_syntax(folder_start);
#ifdef NIX_BASE
	if (!this->has_file(folder_start))
		return std::vector<std::string>{};
#endif

	std::vector<std::string> options = { mod1, mod2, mod3 };
	bool folders = false;
	bool files = false;
	bool recursive = false;

	std::vector<std::string>::iterator iter;
	for (iter = options.begin(); iter != options.end(); ++iter) {
		if (*iter == "folders") {
			folders = true;
		}
		else if (*iter == "files") {
			files = true;
		}
		else if (*iter == "recursive" || *iter == "recurse") {
			recursive = true;
		}
	}
	if (files == 0 && folders == 0) {
		return std::vector<std::string>({ "" });
	}
	std::vector<std::string> track_vec;
	dir_continued(folder_start, track_vec, folders, files, recursive);

#if defined(NIX_BASE)
	for (size_t i = 0; i < track_vec.size(); i++) {
		track_vec[i] = re::sub("/+", "/", track_vec[i]);
	}

#elif defined(WIN_BASE)
	for (size_t i = 0; i < track_vec.size(); i++) {
		track_vec[i] = re::sub("([\\\\]+|\\\\)", "\\", track_vec[i]);
	}
#endif
	return track_vec;
}



std::string OS::pwd() {
#if defined(NIX_BASE)
	char result[FILENAME_MAX];
	ssize_t count = readlink("/proc/self/exe", result, FILENAME_MAX);
	return re::sub("/[^/]*$", "", std::string(result, (count > 0) ? count : 0));

#elif defined(WIN_BASE)
	char buf[256];
	GetCurrentDirectoryA(256, buf);
	return std::string(buf) + '\\';
#endif
}

OS OS::popen(const std::string& command, char leave) {
	// leave styles
	// p = pass (nothing happens = defult)
	// t = throw exception if command fails
	// e = exit if command fails
	const char* commmand_ptr = &command[0];
	m_command.clear();
	int buf_size = 512;
	char buffer[512];

#if defined(NIX_BASE)
	FILE* file = ::popen(commmand_ptr, "r");
#elif defined(WIN_BASE)
	FILE* file = ::_popen(commmand_ptr, "r");
#endif

	while (!feof(file)) {
		if (fgets(buffer, buf_size, file) != NULL)
			m_command += buffer;
	}

#if defined(NIX_BASE)
	int returnCode = pclose(file);
#elif defined(WIN_BASE)
	int returnCode = _pclose(file);
#endif

	if (returnCode) {
		std::cout << "\nConsole Command Failed:\n" << command << std::endl;

		switch (leave) {
		case 't': throw;
		case 'e': exit(1);
		}
	}
	m_last_read = 'c';
	return *this;
}

std::string OS::operator()(const std::string& command) {
	return this->popen(command).read();
}

// ============================================================================================

bool OS::has_file(const std::string& file) { // no '_' based on ord namespace syntax with keyword 'find'

	assert_folder_syntax(file);
	std::ifstream os_file(&file[0]);
	bool file_exists = false;
	if (os_file.is_open()) {
		file_exists = true;
		os_file.close();
	}
	return file_exists;
}


OS OS::move_file(const std::string& old_location, const std::string& new_location) {
	Old_New on = this->id_old_new(old_location, new_location);
	std::ifstream  in(on.old_location, std::ios::in | std::ios::binary);
	std::ofstream out(on.new_location, std::ios::out | std::ios::binary);
	out << in.rdbuf();
#if defined(NIX_BASE)
	::remove(on.old_location.c_str());
#elif defined(WIN_BASE)
	DeleteFileA(on.old_location.c_str());
#endif
	return *this;
}

OS OS::copy_file(const std::string& old_location, const std::string& new_location) {

	std::ifstream  in(old_location, std::ios::in | std::ios::binary);
	std::ofstream out(new_location, std::ios::out | std::ios::binary);
	out << in.rdbuf();
	return *this;
}

OS OS::clear_file(const std::string& content) {
	std::string file_to_wipe = (content == "") ? m_file_name : content;
	this->assert_folder_syntax(file_to_wipe);
	std::ofstream ofs;
	ofs.open(file_to_wipe, std::ofstream::out | std::ofstream::trunc);
	ofs.close();
	return *this;
}


OS OS::delete_file(const std::string& content) {
	std::string file_to_delete = (content == "") ? m_file_name : content;
	this->assert_folder_syntax(file_to_delete);
#if defined(NIX_BASE)
	::remove(&file_to_delete[0]);
#elif defined(WIN_BASE)
	DeleteFileA(file_to_delete.c_str());
#endif
	return *this;
}

OS OS::mkdir(const std::string& folder) {
	this->assert_folder_syntax(folder);
	std::string folder_path = re::sub(R"([\w\d_\s].*$)", "", folder);
	std::vector<std::string> folders = re::split(R"([\\\\/](?=[^\\s]))", re::sub(R"(^(\\.?)[\\\\/]*|[\\\\/]*$)", "", folder));
	std::vector<std::string>::iterator iter;
	for (iter = folders.begin(); iter != folders.end(); ++iter) {
		if (re::match("[a-zA-Z0-9_]*", *iter)) {
			folder_path += '/' + *iter;
			if (this->has_file(folder_path) == false) {

#if defined(NIX_BASE)
				::mkdir(&folder_path[0], S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

#elif defined(WIN_BASE)
				CreateDirectoryA(folder_path.c_str(), NULL);
#endif
			}
		}
	}
	return *this;
}

OS OS::copy_dir(const std::string& old_location, const std::string& new_location) {
	Old_New on = this->id_old_new(old_location, new_location);

	std::string pre_req_folder = re::sub(R"((?:=[^\\/])[^\\/]*$)", "", re::sub(R"([\\/]*$)", "", on.new_location));

	if (!this->has_file(pre_req_folder))
		this->mkdir(pre_req_folder);

	std::vector<std::string> old_folders = this->dir(on.old_location, "recursive", "folders");
	std::vector<std::string> old_files = this->dir(on.old_location, "recursive", "files");

	for (std::string& folder : old_folders) {
		this->mkdir(on.new_location + folder.substr(1, folder.size() - 2));
	}
	for (std::string& file : old_files) {
		this->copy_file(on.new_location + file.substr(1, file.size() - 2));
	}
	return *this;
}

OS OS::move_dir(const std::string& old_location, const std::string& new_location) {
	Old_New on = this->id_old_new(old_location, new_location);

	this->copy_dir(on.old_location, on.new_location);
	this->rmdir(on.old_location);
	return *this;
}

using std::cout; using std::endl;

OS OS::rmdir(const std::string& folder) {
	this->assert_folder_syntax(folder);

	std::vector<std::string> recursive_files = dir(folder, "files", "folders", "recurse");

	int max_slash = -1;
	auto get_max_path = [&recursive_files, &max_slash]() -> void {
		std::for_each(recursive_files.begin(), recursive_files.end(),
			[&max_slash](std::string item) -> void {
			int current_count = re::count(R"([^\\/][\\/])", item);
			if (max_slash < current_count) {
				max_slash = current_count;
			};
		});
	};

	auto rm_slash_count = [&recursive_files, &max_slash]() -> void {
		int count = 0;
		std::vector<std::string> parent_folders;
		std::for_each(recursive_files.begin(), recursive_files.end(),
			[&](std::string item) {

			if (re::count(R"([^\\/][\\/])", item) == max_slash && item.size()) {
				if (re::scan(R"([\\/]$)", item)) {

					parent_folders.push_back(item);

				}
				else {
#if defined(NIX_BASE)
					::remove(&item[0]);
#elif defined(WIN_BASE)
					DeleteFileA(item.c_str());
#endif
				}
				recursive_files[count] = "";
			};

			for (std::string& parent_folder : parent_folders)
#if defined(NIX_BASE)
				::rmdir(&parent_folder[0]);
#elif defined(WIN_BASE)
				RemoveDirectoryA(parent_folder.c_str());
#endif
			count += 1;
		});
	};

	get_max_path();
	rm_slash_count();
	while (max_slash > 1) {
		max_slash = 0;
		get_max_path();
		rm_slash_count();
	}
#if defined(NIX_BASE)
	::rmdir(&folder[0]);
#elif defined(WIN_BASE)
	RemoveDirectoryA(folder.c_str());
#endif

	return *this;
}

void OS::assert_folder_syntax(const std::string& folder1, const std::string& folder2) {
	assert(re::match(R"(^[\\.[\\/]+[\w\d_/\\\.]*$)", folder1));
	assert(!re::scan(R"([^\\]\s)", folder1));
	if (folder2.size()) {
		assert(re::match(R"(^[\\.[\\/]+[\w\d_/\\\.]*$)", folder2));
		assert(!re::scan(R"([^\\]\s)", folder2));
	}
}

OS::Old_New OS::id_old_new(std::string old_location, std::string new_location) {
	if (!new_location.size()) {
		new_location = old_location;
		old_location = m_file_name;
	}
	this->assert_folder_syntax(old_location, new_location);

	Old_New on;
	on.old_location = old_location;
	on.new_location = new_location;
	return on;
}

// <<<< file managment
// ============================================================================================
// -------------------------------------------------------------------------------
std::string OS::file_data() { return m_file_data; }
std::string OS::file_name() { return m_file_name; }

std::string OS::command() { return m_command; }
std::string OS::read_command() { return m_command; }
std::string OS::cmd() { return m_command; }
// -------------------------------------------------------------------------------
// ============================================================================================

OS os;