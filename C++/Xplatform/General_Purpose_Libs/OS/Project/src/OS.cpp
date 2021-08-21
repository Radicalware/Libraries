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

// -------------------------------------------------------------------------------------------
#if defined(NIX_BASE)
void OS::Dir_Continued(const xstring scan_start, xvector<xstring>& track_vec, \
    const bool folders, const bool files, const bool recursive)
{
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
                    OS::Dir_Continued(scan_start + "/" + dir_item, \
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
void OS::Dir_Continued(const xstring folder_start, xvector<xstring>& track_vec, \
    const bool folders, const bool files, const bool recursive)
{

    WIN32_FIND_DATAA find_data;
    //LPWIN32_FIND_DATAA find_data = nullptr;
    //TCHAR dir_size_tchar[MAX_PATH];
    char dir_size_char[MAX_PATH];
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // If the directory is not specified as a command-line argument,
    // print usage.

    //std::wstring wstr(item.begin(), item.end());

    if (folder_start.size() >= MAX_PATH - 3) {
        std::cerr << "OS::Dir_Continued >> (folder_start.size() >= MAX_PATH - 3) >> should be false";
        return;
    }

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
        std::string tmp_path;
        xstring Full_Path;
        do
        {
            //if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                file_path_str = t_path_to_str_path() + "\\";
                if (file_path_str == ".\\" || file_path_str == "..\\")
                    continue;
                if (*(folder_start.end() - 1) == '\\')
                    Full_Path = folder_start + file_path_str;
                else
                    Full_Path = folder_start + '\\' + file_path_str;


                if (folders)
                    track_vec.push_back(Full_Path);
                if (recursive)
                    Dir_Continued(Full_Path, track_vec, folders, files, recursive);
            }
            else {
                if (files) {
                    tmp_path = t_path_to_str_path();
                    if (*(folder_start.end() - 1) == '\\' || tmp_path[0] == '\\')
                        track_vec.push_back(folder_start + t_path_to_str_path());
                    else
                        track_vec.push_back(folder_start + '\\' + t_path_to_str_path());
                }
            }
            //} while (FindNextFile(hFind, &find_data) != 0);
        } while (FindNextFileA(hFind, &find_data) != 0);

        FindClose(hFind);
    }
};
#endif

// -------------------------------------------------------------------------------------------

OS::OS()
{
    m_last_read = 'n';
};

OS::~OS() {
    File.Close();
};

xvector<int> OS::GetConsoleSize() // [columns, rows]
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))

    CONSOLE_SCREEN_BUFFER_INFO screen_info;

    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &screen_info);
    return xvector<int>{
        screen_info.srWindow.Right - screen_info.srWindow.Left + 1, // columns
            screen_info.srWindow.Bottom - screen_info.srWindow.Top + 1  // rows
    };
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return xvector<int>({ w.ws_col, w.ws_row });
#endif
}

bool OS::HasFileSyntax(const xstring& i_file) {
    if (!i_file.Match(R"(^([\./\\]+?)[\-\d\w\.\\/]+$)"))
        return false;
    else if (i_file.Scan(R"([^\\]\s)"))
        return false;
    else
        return true;
}

bool OS::AllHaveFileSyntax(const xvector<xstring>& i_files)
{
    for (const xstring& i_file : i_files) {
        if (!(OS::HasFileSyntax(i_file)))
            return false;
    }return true;
}


// ---------------------------------------------------------------------------------------------

void OS::Touch(const xstring& new_file)
{
    if (new_file.size())
        File.m_name = OS::FullPath(new_file);

    if (OS::Has(File.m_name))
        return;

    File.SetWrite();
}


void OS::MKDIR(const xstring& folder)
{
    if (!folder.size()) return;

    xstring new_folder = OS::FullPath(folder);

    xvector<xstring> folder_names = new_folder.Split(R"([\\/](?=[^\\/]|$))");
    //std::cout << '\n';
    //std::cout << "orig:  " << folder << '\n';
    //std::cout << "path:  " << new_folder << '\n';
    //std::cout << "split: " << folder_names.Join('-') << '\n';
    xstring folder_path;
    for (xvector<xstring>::iterator iter = folder_names.begin(); iter < folder_names.end(); ++iter)
    {
        if (!iter->size()) continue;
       // std::cout << "iter = " << folder_path << '\n';
#if defined(NIX_BASE)
        folder_path += '/' + *iter;
        ::mkdir(folder_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

#elif defined(WIN_BASE)
        if ((*iter)[1] == ':')
        {
            folder_path += *iter;
            continue;
        }
        folder_path += '\\' + *iter;
        CreateDirectoryA(folder_path.c_str(), NULL);
#endif
    }
}


void OS::CP(const xstring& old_location, const xstring& new_location)
{
    if (!(old_location.size() || new_location.size())) return;

    xstring old_path = OS::FullPath(old_location);
    xstring new_path = OS::FullPath(new_location);

    if (OS::HasFile(old_path))
        OS::CopyFile(old_path, new_path);
    else if (OS::HasDir(new_path))
        OS::CopyDir(old_path, new_path);
    else
        throw std::runtime_error("Location of Copy Not Found: " + old_location);
}


void OS::MV(const xstring& old_location, const xstring& new_location)
{
    xstring old_path = OS::FullPath(old_location);
    xstring new_path = OS::FullPath(new_location);

    if (!(old_location.size() || new_location.size())) return;

    if (OS::HasFile(old_path))
        OS::MoveFile(old_path, new_path);
    else if (OS::HasDir(old_path))
        OS::MoveDir(old_path, new_path);
    else
        throw std::runtime_error("Move Start Location Not Found: " + old_location);
}


void OS::RM(const xstring& del_file)
{
    if (!del_file.size()) return;

    xstring bad_file = OS::FullPath(del_file);

    if (OS::HasFile(bad_file))
        OS::RemoveFile(bad_file);

    else if (OS::HasDir(bad_file))
        OS::RemoveDir(bad_file);
}

// ---------------------------------------------------------------------------------------------

OS OS::Open(const xstring& new_file_name, const char write_method)
{
    // r = read       
    // w = write mode 
    // a = append

    File.Close();
    File.SetFile(new_file_name);

    switch (write_method)
    {
    case 'r':
        File.SetRead();
        break;

    case 'w':
        File.SetWrite();
        break;

    case 'a':
        File.SetAppend();
        break;

    default:
        File.SetRead();
    }

    m_last_read = 'f'; // 'f' for file opposed to 'c' for command
    return *this;
}

OS OS::Close()
{
    File.Close();
    return *this;
}

// os.Open(file_name).Read()
// os.RunConsoleCommand(command).Read()
xstring OS::Read(char content, bool close_file /* = false */)
{
    content = (content == 'd') ? m_last_read : content;
    switch (content) {
    case 'f':  return this->InstRead();
    case 'c':  return CMD.m_out;
    default:  return xstring("none");
    }
    if (close_file)
        this->Close();
}

xstring OS::InstRead()
{
    if (File.m_handler != 'r')
        File.SetRead();

    File.m_data.clear();
    xstring line;
    errno = 0;
    if (File.m_in_stream.is_open()) {
        while (getline(File.m_in_stream, line))
            File.m_data += line + '\n';
    }
    else {
        xstring err("Error (" + ToXString(errno) + "): Could Not Open Text File: ");
        err += File.m_name;
        throw std::runtime_error(err.c_str());
    }
    return File.m_data;
}


xstring OS::ReadFastMethod(const xstring& file_name, bool re_try)
{
    if (!file_name.size())
        return xstring();

    errno = 0;
    FILE* fp;
#if defined(WIN_BASE)
    fopen_s(&fp, file_name.c_str(), "rb");
#else
    fp = fopen(file_name.c_str(), "rb");
#endif
    if (fp == nullptr && !re_try) {
        xstring err("Error (" + ToXString(errno) + "): Could Not Open Text File: ");
        err += file_name;
        throw std::runtime_error(err.c_str());
    }
    else if ((errno || fp == nullptr) && re_try)
    {
        if (fp) fclose(fp);
        return OS::ReadStatMethod(file_name);
    }

    fseek(fp, 0, SEEK_END);
    const long int file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (!file_size) {
        if (fp) fclose(fp);
        return xstring();
    }

    //char* rbuffer = static_cast<char*>(malloc(file_size + 1 * sizeof(char)));
    unsigned char* rbuffer = static_cast<unsigned char*>(calloc(file_size, sizeof(unsigned char) + 1));
    if (rbuffer != nullptr) {
#if defined(WIN_BASE)
        fread(rbuffer, sizeof(unsigned char), file_size, fp);
#else
        size_t unused = fread(rbuffer, sizeof(unsigned char), file_size, fp); // -Wunused-result would not work
#endif
    }
    else {
        if (rbuffer) free(rbuffer); // I know. . . but I want it here anyway.
        if (fp) fclose(fp);
        throw std::runtime_error("Failed to allocated file data buffer\n");
    }

    xstring rets;
    if (rbuffer)
    {
        rets = rbuffer;
        free(rbuffer);
    }
    if (fp) fclose(fp);

    return rets;
}

xstring OS::ReadStatMethod(const xstring& file_name)
{

    if (!file_name.size())
        return xstring();

    struct stat stat_buf;
    int state = stat(file_name.c_str(), &stat_buf);
    size_t file_size = stat_buf.st_size;
    if (file_size < 1)
        return xstring();

    FILE* fp;
#if defined(WIN_BASE)
    fopen_s(&fp, file_name.c_str(), "rb");
#else
    fp = fopen(file_name.c_str(), "rb");
#endif
    unsigned char* rbuffer = static_cast<unsigned char*>(calloc(file_size, sizeof(unsigned char)));
    errno = 0;
    if (rbuffer != nullptr && fp != nullptr)
        fread(rbuffer, sizeof(char), file_size, fp);
    else {
        if (rbuffer) free(rbuffer);
        if (fp)fclose(fp);
        throw std::runtime_error("Could not allocate stat_read buffer's ByteArray!\n");
    }

    if (fp) fclose(fp);
    xstring rets;
    rets.insert(rets.begin(), rbuffer, rbuffer + file_size);
    if (rbuffer) free(rbuffer);
    return rets;
}

xstring OS::ReadStreamMethod(const xstring& file_name)
{

    if (!file_name.size())
        return xstring();

    std::ifstream os_file(file_name, std::ios::in | std::ios::binary); // ios::in for data intake 

    xstring file_data;
    xstring line;

    errno = 0;
    if (os_file.is_open()) {
        while (getline(os_file, line))
            file_data += line + '\n';
        os_file.close();
    }
    else {
        xstring err("Error (" + ToXString(errno) + "): Could Not Open Text File: ");
        err += file_name;
        throw std::runtime_error(err.c_str());
    }
    return file_data;
}

// you must use open before write
OS OS::Write(const xstring& content, bool store /* = false */)
{
    if (File.m_handler == 'a')
        File.SetAppend();
    else
        File.Clear();

    errno = 0;
    if (!File.m_out_stream.is_open())
        throw std::runtime_error("Error (" + ToXString(errno) + ") Unable to Open File: " + File.m_name + "\n");

    if (store)
        File.m_data = content;
    else
        File.m_data.clear();

    File.m_out_stream << content;

    m_last_read = 'f';
    return *this;
}


xvector<xstring> OS::Dir(const xstring& folder_start, const char mod1, const char mod2, const char mod3)
{
    // recursive = search foldres recursivly
    // folders   = return folders in search
    // files     = return files in search

    if (!folder_start.size()) return xvector<xstring>();

    xstring search_path = OS::FullPath(folder_start);
    xvector<xstring> track_vec;

    if (folder_start == "" || (!OS::HasDir(search_path))) {
        return track_vec;
    }

    const char* options = nullptr;
    options = new const char[4]{ mod1, mod2, mod3, '\0' };

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
    Dir_Continued(search_path, track_vec, directories, files, recursive);
    delete[] options;

    for (xstring& str : track_vec)
        str.RemoveNulls();

    return track_vec;
}

OS& OS::RunConsoleCommand(const xstring& command, char leave)
{
    // leave styles
    // p = pass (nothing happens = defult)
    // t = throw exception if command fails
    // e = exit if command fails

#if defined(NIX_BASE)
    CMD.m_cmd = command + " 2>&1";
#elif defined(WIN_BASE)
    CMD.m_cmd = command;
#endif

    CMD.m_out.clear();
    int buf_size = 512;
    char buffer[512];

#if defined(NIX_BASE)
    std::cout << cmd.m_cmd.c_str() << std::endl;
    FILE* file = popen(cmd.m_cmd.c_str(), "r");
#elif defined(WIN_BASE)
    FILE* file = _popen(CMD.m_cmd.c_str(), "r");
#endif

    while (!feof(file)) {
        if (fgets(buffer, buf_size, file) != NULL)
            CMD.m_out += buffer;
    }

#if defined(NIX_BASE)
    int returnCode = pclose(file);
#elif defined(WIN_BASE)
    int returnCode = _pclose(file);
#endif

    if (returnCode) {
        CMD.m_err = CMD.m_out;
        CMD.m_err_message = xstring("\nConsole Command Failed:\n") + CMD.m_cmd + "\nerror out: " + CMD.m_err;
        if (leave == 'd') {
            m_last_read = 'c';
            return *this;
        }
        switch (leave) {
        case 't': throw std::runtime_error(CMD.m_err);                  // t = throw error
        case 'x': std::cout << CMD.m_err_message << std::endl; exit(1); // x = eXit
        case 'm': std::cout << CMD.m_err_message << std::endl; break;   // m = full error Message
        case 'e': std::cout << CMD.m_err << std::endl; break;           // e = standard Error
        case 'p': std::cout << CMD.m_err << std::endl; break;           // p = Programatic error
        }
    }
    else {
        CMD.m_err.clear();
        CMD.m_err_message.clear();
    }
    m_last_read = 'c';
    return *this;
}

xstring OS::operator()(const xstring& command, const char leave) {
    return this->RunConsoleCommand(command, leave).Read();
};

// ============================================================================================

void OS::AssertFolderSyntax(const xstring& folder)
{
    if (!folder.Match(R"(^([\./\\]+?)[\-\d\w\.\\/]+$)")) {
        throw std::runtime_error("Failed Dir Syntax = "
            R"(^([\./\\]+?)[\-\d\w\.\\/]+$)"
            "\n  what():  Dir Item: " + folder + "\n");
    }

    if (folder.Scan(R"([^\\]\s)")) {
        throw std::runtime_error("You can't have a space in a dir item\n" \
            "  what():  without an escape char\n");
    }
}

void OS::MoveFile(const xstring& old_location, const xstring& new_location)
{
    if (!(old_location.size() || new_location.size())) return;

    xstring old_path = OS::FullPath(old_location);
    xstring new_path = OS::FullPath(new_location);
    try {
        std::ifstream  in(old_path, std::ios::in | std::ios::binary);
        std::ofstream out(new_path, std::ios::out | std::ios::binary);
        out << in.rdbuf();

        if (in.is_open()) in.close();
        if (out.is_open()) out.close();
    }
    catch (std::runtime_error & err) {
        throw std::runtime_error(xstring("OS::move_file Failed: ") + err.what());
    }

    OS::RemoveFile(old_path);
}

void OS::MoveDir(const xstring& old_location, const xstring& new_location) {

    if (!(old_location.size() || new_location.size())) return;

    OS::CopyDir(old_location, new_location);
    OS::RemoveDir(old_location);
}


void OS::CopyFile(const xstring& old_location, const xstring& new_location)
{
    if (!(old_location.size() || new_location.size())) return;

    xstring old_path = OS::FullPath(old_location);
    xstring new_path = OS::FullPath(new_location);

    std::ifstream  in(old_path, std::ios::in | std::ios::binary);
    std::ofstream out(new_path, std::ios::out | std::ios::binary);

    out << in.rdbuf();
    if (in.is_open()) in.close();
    if (out.is_open()) out.close();
}

void OS::CopyDir(const xstring& old_location, const xstring& new_location)
{

    if (!(old_location.size() || new_location.size()))
        return;

    xstring old_path = OS::FullPath(old_location);
    xstring new_path = OS::FullPath(new_location);

    if (!OS::HasDir(old_path))
        throw std::runtime_error("\nThis is not a folder\n" + old_path + '\n');


    if (OS::HasFile(new_path))
        throw std::runtime_error("\nA file exists there\n" + new_path + '\n');

    xvector<xstring> old_folders = OS::Dir(old_path, 'r', 'd');
    xvector<xstring> old_files = OS::Dir(old_path, 'r', 'f');
    size_t old_size = old_path.size();

    xstring nested_dir_item;
    if (old_folders.size()) {
        for (xvector<xstring>::const_iterator it = old_folders.begin(); it < old_folders.end(); it++) {
            nested_dir_item = (*it).substr(old_size, (*it).size() - old_size);
            OS::MKDIR(new_path + nested_dir_item);
        }
    }
    else {
        OS::MKDIR(new_path);
    }

    if (old_files.size()) {
        for (xvector<xstring>::const_iterator it = old_files.begin(); it < old_files.end(); it++) {
            nested_dir_item = (*it).substr(old_size, (*it).size() - old_size);
            OS::CopyFile(*it, new_path + nested_dir_item);
        }
    }
}

void OS::RemoveFile(const xstring& item)
{
    xstring fitem = OS::FullPath(item);

    if (!OS_O::Dir_Type::HasFile(fitem))
        throw std::runtime_error(std::string("Error Filename: ") + fitem + " Does Not Exist!\n");

    errno = 0;
    try {

#if   defined(NIX_BASE)
        remove(fitem.c_str());
#elif defined(WIN_BASE)
        DeleteFileA(fitem.c_str());
#endif
        if (errno)
            throw;
    }
    catch (...) {
        xstring err = "Failed (" + ToXString(errno) + "): Failed to delete file: '" + fitem + "'\n";
        throw std::runtime_error(err);
    }
}


void OS::RemoveDir(const xstring& folder)
{
    if (!folder.size()) return;

    xstring del_path = OS::FullPath(folder);

    xvector<xstring> dir_items = OS::Dir(del_path, 'r', 'f', 'd');
    dir_items.push_back(del_path);

    std::multimap<size_t, xstring> dir_size_mp; // I only have a unordered_map of the extended stl at this time

    for (xvector<xstring>::const_iterator it = dir_items.begin(); it != dir_items.end(); it++)
        dir_size_mp.insert({ it->Count("([\\\\/][^\\\\/\\s])"), *it });

    std::multimap<size_t, xstring>::const_reverse_iterator dir_item;

    auto delete_dir_item = [&dir_item]() -> void
    {
#if defined(NIX_BASE)
        if (OS::HasDir(dir_item->second))
            ::rmdir(dir_item->second.c_str());
        else if (OS::HasFile(dir_item->second))
            remove(dir_item->second.c_str());

#elif defined(WIN_BASE)
        if (OS::HasDir(dir_item->second))
            RemoveDirectoryA(dir_item->second.c_str());
        else if (OS::HasFile(dir_item->second))
            DeleteFileA(dir_item->second.c_str());

#endif
    };

    for (dir_item = dir_size_mp.rbegin(); dir_item != dir_size_mp.rend(); dir_item++)
        delete_dir_item();
}

void OS::ClearFile(const xstring& i_file)
{
    if (!i_file.size())
        return;

    xstring file = OS::FullPath(i_file);
    std::ofstream ofs;
    ofs.open(file.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (ofs.is_open())
        ofs.close();
}

// <<<< file managment ------------------------------------------------------------------------
// ============================================================================================

