#pragma warning ( disable : 26444) // Allow un-named objects
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
* WITHOUT WARRANTIES OR CONDITIONS Ofls ANY KIND, either express or implied.
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
#include "dir_support/Dir_Type.h"
#include "dir_support/File_Names.h"


// -------------------------------------------------------------------------------------------
#if defined(NIX_BASE)
void OS::dir_continued(const xstring scan_start, xvector<xstring>& track_vec, \
	const bool folders, const bool files, const bool recursive) {

	DIR* current_dir = opendir(scan_start.c_str());
	xstring dir_item;
	// starting dir given as a xstring
	// "dir_item" can be anything in the "current_dir" 
	// such as a new folder, file, binary, etc.
	while (struct dirent* dir_item_ptr = readdir(current_dir)) {
		dir_item = (dir_item_ptr->d_name);
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
void OS::dir_continued(const xstring folder_start, xvector<xstring>& track_vec, \
	const bool folders, const bool files, const bool recursive) {

	WIN32_FIND_DATA find_data;
	//LPWIN32_FIND_DATAA find_data = nullptr;
	//TCHAR dir_size_tchar[MAX_PATH];
	char dir_size_char[MAX_PATH];
	HANDLE hFind = INVALID_HANDLE_VALUE;

	// If the directory is not specified as a command-line argument,
	// print usage.

	//std::wstring wstr(item.begin(), item.end());
	assert(folder_start.size() < MAX_PATH - 3);

	// Prepare string for use with FindFile functions.  First, copy the
	// string to a buffer, then append '\*' to the directory name.

	// StringCchCopy(dir_size_tchar, MAX_PATH, wstr.c_str());
	// StringCchCat(dir_size_tchar, MAX_PATH, TEXT("\\*"));
	StringCchCopyA(dir_size_char, MAX_PATH, folder_start.c_str());
	StringCchCatA(dir_size_char, MAX_PATH, "\\*");

	// Find the first file in the directory.
	// hFind = FindFirstFile(dir_size_tchar, &find_data);
	hFind = FindFirstFileA(dir_size_char, &find_data);

	auto t_path_to_str_path = [&find_data]() -> xstring {
		xstring path_str;
		int count = 0;
		// while (find_data.cFileName[count] != '\0') {
		//     path_str += find_data.cFileName[count];
		//     ++count;
		// }
		while (find_data.cFileName[count] != '\0') {
			path_str += find_data.cFileName[count];
			++count;
		}
		return path_str;
	};

	// check to make sure it is there
	if (INVALID_HANDLE_VALUE != hFind) {
		// List all the files in the directory with some info about them.
		xstring file_path_str;
		do
		{
			//if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
				file_path_str = t_path_to_str_path() + "\\";
				if (file_path_str == ".\\" || file_path_str == "..\\")
					continue;
				xstring full_path = folder_start + "\\" + file_path_str;
				if (folders)
					track_vec.push_back(full_path);
				if (recursive)
					dir_continued(full_path, track_vec, folders, files, recursive);
			}
			else {
				if (files)
					track_vec.push_back(folder_start + "\\" + t_path_to_str_path());
			}
			//} while (FindNextFile(hFind, &find_data) != 0);
		} while (FindNextFileA(hFind, &find_data) != 0);

		FindClose(hFind);
	}
};
#endif

// -------------------------------------------------------------------------------------------

OS::OS() {
	m_command = "";
	m_std_out = "";
	m_std_err = "";
	m_err_message = "";

	m_file_name = "";
	m_file_data = "";

	m_last_read = 'n';
	m_write_method = 'a';

	m_rexit = false;

};

OS::~OS() {
};

xvector<int> OS::console_size() // [columns, rows]
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))

	CONSOLE_SCREEN_BUFFER_INFO screen_info;

	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &screen_info);
	return xvector<int>{
		screen_info.srWindow.Right - screen_info.srWindow.Left + 1, // columns
		screen_info.srWindow.Bottom - screen_info.srWindow.Top + 1  // rows
	};
#else
	return xvector<int>();
#endif
}

void OS::set_file_regex(bool rexit) {
    m_rexit = rexit;
}

bool OS::file_regex_status() { return m_rexit; }

bool OS::file_syntax(const xstring& i_file) {
    if (!i_file.match(R"(^([\./\\]+?)[\-\d\w\.\\/]+$)"))
        return false;
	else if (i_file.scan(R"([^\\]\s)"))
        return false;
    else 
		return true;
}

bool OS::file_list_syntax(const xvector<xstring>& i_files) {
    for (const xstring& i_file : i_files) {
        if (!(file_syntax(i_file))) 
            return false;
    }return true;
}

// ---------------------------------------------------------------------------------------------

void OS::touch(const xstring& new_file) {

	if (new_file.size())
		m_file_data = new_file;

	if (m_file_data.empty())
		return;

	File_Names fls(m_rexit, new_file);
    std::ofstream os_file(fls.target().c_str());
    if (os_file.is_open()) os_file.close();
}


void OS::cp(const xstring& old_location, const xstring& new_location) {
	File_Names fls(m_rexit, old_location, new_location);

	if (!(old_location.size() || new_location.size())) return;

    if (this->file(fls.old())) 
        this->copy_file(fls.old(), fls.target());
    else if (this->directory(fls.old())) 
        this->copy_dir(fls.old(), fls.target());
    else 
        throw std::runtime_error("Location of Copy Not Found: " + fls.old());
}


void OS::mv(const xstring& old_location, const xstring& new_location) {
	File_Names fls(m_rexit, old_location, new_location);

	if (!(old_location.size() || new_location.size())) return;

    if (this->has(fls.old()) == os_file) 
        this->move_file(fls.old(), fls.target());
    else if (this->has(fls.old()) == os_directory)
        this->move_dir(fls.old(), fls.target());
    else
        throw std::runtime_error("Move Start Location Not Found: " + fls.old());
}


void OS::rm(const xstring& del_file) {
	if (!del_file.size()) return;

	File_Names fls(m_rexit, del_file);

    if (this->file(fls.target()))
		this->delete_file(fls.target());

    else if (this->directory(fls.target()))
		this->delete_dir(fls.target());
}

// ---------------------------------------------------------------------------------------------

OS OS::open(const xstring& new_file_name, const char write_method) {
    // a = append     (append then writes like in python)
    // w = write mode (clears then writes like in python)

    //File_Names fls(m_rexit, new_file_name);
    m_write_method = write_method;
    //m_file_name = fls.target(); 

#ifdef WIN_BASE
	char full[_MAX_PATH];
#else
	char full[PATH_MAX];
#endif
#if defined(WIN_BASE)
	 _fullpath(full, new_file_name.sub("/", "\\\\").c_str(), _MAX_PATH);
#elif defined(NIX_BASE)
	realpath(new_file_name.sub("\\\\", "/").c_str(), full);
#endif

	m_file_name.clear();
	m_file_name = full; // TODO: fix, this causes an issue in linux when global OS is set in the h file
	// FIX: Just don't set the extern OS in the header, instead set it in the main.cpp

	if (m_write_method == 'w')
		this->clear_file(m_file_name);

    m_last_read = 'f';
    return *this;
}

// os.open(file_name).read()
// os.popen(command).read()
xstring OS::read(char content) {
    content = (content == 'n') ? m_last_read : content;
    switch (content) {
		case 'f':  return this->read_file();
		case 'c':  return m_std_out;
		default:  return xstring("none");
    }

}


xstring OS::read_file() {
    std::ifstream os_file(m_file_name);
    xstring line;
    m_file_data.clear();

    if (os_file.is_open()) {
        while (getline(os_file, line)) {
			m_file_data += line + '\n';
        }
        os_file.close();
    }
    return m_file_data;
}


OS OS::write(const xstring& content, char write_method) {


	if (!m_file_name.size()) return *this;

    m_write_method = (write_method == 'n') ? m_write_method : write_method;

    switch (m_write_method) {
    case 'a':  m_write_method = 'a';
        m_file_data = this->read() + content;
        break;

    case 'w':  m_write_method = 'w';
		this->clear_file(m_file_name);
        m_file_data = content.c_str();
    }
    std::ofstream os_file;
    os_file.open(m_file_name);
    if (os_file.is_open()) {
        os_file << m_file_data;
        os_file.close();
    }

    m_last_read = 'f';
    return *this;
}


xvector<xstring> OS::dir(const xstring& folder_start, const char mod1, const char mod2, const char mod3) {
    // recursive = search foldres recursivly
    // folders   = return folders in search
    // files     = return files in search

	if (!folder_start.size()) return xvector<xstring>();

	File_Names fls(m_rexit, folder_start);
	xvector<xstring> track_vec;

    if (folder_start == "" || (!this->directory(fls.target()))) {
        return track_vec;
    }

   const char* options = new const char[4]{ mod1, mod2, mod3, '\0'};

    bool files = false;
    bool directories = false;
    bool recursive = false;

    auto set_mods = [&files, &directories, &recursive](const char option) -> void {
        switch (option) {
			case 'f':files = true; break;
            case 'd':directories = true; break;
			case 'r':recursive = true;
        }
    };
    for (int i = 0; i < 3; i++)
        set_mods(options[i]);

    if (files == 0 && directories == 0) {
        return xvector<xstring>{ "" };
    }
    dir_continued(fls.target(), track_vec, directories, files, recursive);
	delete [] options;
    return track_vec;
}

OS OS::popen(const xstring& command, char leave) {
    // leave styles
    // p = pass (nothing happens = defult)
    // t = throw exception if command fails
    // e = exit if command fails

#if defined(NIX_BASE)
    m_command = command + xstring(" 2>&1");
#elif defined(WIN_BASE)
    m_command = command;
#endif

    m_std_out.clear();
    int buf_size = 512;
    char buffer[512];

#if defined(NIX_BASE)
    FILE* file = ::popen(m_command.c_str(), "r");
#elif defined(WIN_BASE)
    FILE* file = ::_popen(m_command.c_str(), "r");
#endif

    while (!feof(file)) {
        if (fgets(buffer, buf_size, file) != NULL)
            m_std_out += buffer;
    }

#if defined(NIX_BASE)
    int returnCode = pclose(file);
#elif defined(WIN_BASE)
    int returnCode = _pclose(file);
#endif

    if (returnCode) {
        m_std_err = m_std_out;
        m_err_message = xstring("\nConsole Command Failed:\n") + m_command + "\nerror out: " + m_std_err;
        switch (leave) {
            case 't': throw std::runtime_error (m_std_err);
            case 'x': exit(1);
            case 'm': std::cout << m_err_message << std::endl; break;
            case 'e': std::cout << m_std_err << std::endl; break;
            default:  std::cout << m_err_message << std::endl; break;
        }
    } else {
        m_std_err.clear();
        m_err_message.clear();
    }
    m_last_read = 'c';
    return *this;
}

xstring OS::operator()(const xstring& command) {
	return this->popen(command).read();
};

template<typename T>
void OS::p(const T& input){
    std::cout << input << std::endl;
}

template<typename T, typename X>
void OS::p(const T& input1, const X& input2){
    std::cout << input1 << input2 << std::endl;
}


// ============================================================================================

void OS::move_file(const xstring& old_location, const xstring& new_location) {


	if (!(old_location.size() || new_location.size())) return;

	File_Names fls(m_rexit, old_location, new_location);
    std::ifstream  in(fls.old(), std::ios::in | std::ios::binary);
    std::ofstream out(fls.target(), std::ios::out | std::ios::binary);
    out << in.rdbuf();

    if (in.is_open()) in.close();
    if (out.is_open()) out.close();
    
#if defined(NIX_BASE)
    ::remove(fls.old().c_str());
#elif defined(WIN_BASE)
    DeleteFileA(fls.old().c_str());
#endif
}


void OS::copy_file(const xstring& old_location, const xstring& new_location) {

	if (!(old_location.size() || new_location.size())) return;

	File_Names fls(m_rexit, old_location, new_location);
    std::ifstream  in(fls.old(), std::ios::in | std::ios::binary);
    std::ofstream out(fls.target(), std::ios::out | std::ios::binary);

    out << in.rdbuf();
	if (in.is_open()) in.close();
	if (out.is_open()) out.close();
}


void OS::clear_file(const xstring& i_file) {

	if (!i_file.size()) return;

	File_Names fls(m_rexit, i_file);
    std::ofstream ofs;
    ofs.open(fls.target().c_str(), std::ofstream::out | std::ofstream::trunc);
	if (ofs.is_open()) ofs.close();
}


void OS::delete_file(const xstring& item) {

	//if (!item.size()) return;

	//File_Names fls(m_rexit, item);

#if defined(NIX_BASE)
    ::remove(item.c_str());
#elif defined(WIN_BASE)
    DeleteFileA(item.c_str());
#endif
}


void OS::mkdir(const xstring& folder) {

	if (!folder.size()) return;

	File_Names fls(m_rexit, folder);
    //fls.imaginary_path();
    
    xstring additional_dirs = fls.target().substr(this->pwd().size(), fls.target().size() - this->pwd().size());

    xvector<xstring> folder_names = additional_dirs.split(R"([\\/](?=[^\s]))");

    xstring folder_path = this->pwd();
    for (xvector<xstring>::iterator iter = folder_names.begin(); iter < folder_names.end(); ++iter) {
		if (!iter->size()) continue;
#if defined(NIX_BASE)
        folder_path += '/' + *iter;
        ::mkdir(folder_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

#elif defined(WIN_BASE)
        folder_path += '\\' + *iter;
        CreateDirectoryA(folder_path.c_str(), NULL);
#endif
    }
}



void OS::copy_dir(const xstring& old_location, const xstring& new_location) {


	if (!(old_location.size() || new_location.size())) return;

	File_Names fls(m_rexit, old_location, new_location);
	if (!this->directory(fls.old()))
		throw std::runtime_error("\nThis is not a folder\n" + fls.old() + '\n');


	if (this->file(fls.target()))
		throw std::runtime_error("\nA file exists there\n" + fls.target() + '\n');

	xvector<xstring> old_folders = this->dir(fls.old(), 'r', 'd');
	xvector<xstring> old_files = this->dir(fls.old(), 'r', 'f');

	xstring nested_dir_item;
	if (old_folders.size()){
		for (xvector<xstring>::const_iterator it = old_folders.begin(); it < old_folders.end(); it++) {
			nested_dir_item = (*it).substr(fls.old().size(), (*it).size() - fls.old().size());
			this->mkdir(fls.target() + nested_dir_item);
		}
	}else {
		this->mkdir(fls.target());
	}

	if (old_files.size()) {
		for (xvector<xstring>::const_iterator it = old_files.begin(); it < old_files.end(); it++) {
			nested_dir_item = (*it).substr(fls.old().size(), (*it).size() - fls.old().size());
			this->copy_file(*it, fls.target() + nested_dir_item);
		}
	}

}


void OS::move_dir(const xstring& old_location, const xstring& new_location) {

	if (!(old_location.size() || new_location.size())) return;

    this->copy_dir(old_location, new_location);
    this->delete_dir(old_location);
}


void OS::delete_dir(const xstring& folder) {

	if (!folder.size()) return;

    File_Names fls(m_rexit, folder);

    xvector<xstring> dir_items = this->dir(fls.target(), 'r','f','d');
    dir_items.push_back(fls.target());
    xvector<unsigned long long> file_sizes;

    for(size_t i = 0; i < dir_items.size(); i++){
        file_sizes.push_back(dir_items[i].count("([\\\\/][^\\\\/\\s])"));
    }
    if(file_sizes.size())
        std::sort(file_sizes.rbegin(), file_sizes.rend());

    xvector<unsigned long long>::iterator sz;
    xvector<xstring>::iterator dir_item;

    auto delete_dir_item = [&dir_item, this]() -> void {
        #if defined(NIX_BASE)
            if(this->directory(*dir_item)){
                ::rmdir((*dir_item).c_str());
            }else if(this->file(*dir_item)){
                ::remove((*dir_item).c_str());
            }
        #elif defined(WIN_BASE)
            if(this->directory(*dir_item)){
                RemoveDirectoryA((*dir_item).c_str());
            }else if(this->file(*dir_item)){
                DeleteFileA((*dir_item).c_str());
            }
        #endif
    };

    for(sz = file_sizes.begin(); sz < file_sizes.end(); sz++){

        for(dir_item = dir_items.begin(); dir_item < dir_items.end(); dir_item++ ){

            if(*sz == dir_item->count("([\\\\/][^\\\\/\\s])")){
                delete_dir_item();
            }
        }
    }
}

// <<<< file managment ------------------------------------------------------------------------
// ============================================================================================
// >>>> Getters -------------------------------------------------------------------------------
xstring OS::file_data() { return m_file_data; }
xstring OS::file_name() { return m_file_name; }
// -------------------------------------------------------------------------------
xstring OS::cli_command() { return m_command; }

xstring OS::std_in() { return m_command;  }
xstring OS::std_out() { return m_std_out; }
xstring OS::std_err() { return m_std_err;  }

xstring OS::err_message() { return m_err_message;  }
// <<<< Getters -------------------------------------------------------------------------------
// ============================================================================================

extern OS os;
