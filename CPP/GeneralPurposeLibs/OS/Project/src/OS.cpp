﻿#pragma warning ( disable : 26444) // Allow un-named objects
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
#if defined(BxWindows)
void RA::OS::Dir_Continued(const xstring folder_start, xvector<xstring>& track_vec, \
    const bool folders, const bool files, const bool recursive)
{
    Begin();
    WIN32_FIND_DATAA LoFindData;
    //LPWIN32_FIND_DATAA LoFindData = nullptr;
    //TCHAR dir_size_tchar[MAX_PATH];
    char dir_size_char[MAX_PATH];
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // If the directory is not specified as a command-line argument,
    // print usage.

    //std::wstring wstr(item.begin(), item.end());

    if (folder_start.size() >= MAX_PATH - 3) {
        std::cerr << "RA::OS::Dir_Continued >> (folder_start.size() >= MAX_PATH - 3) >> should be false";
        return;
    }

    // Prepare string for use with FindFile functions.  First, copy the
    // string to a buffer, then append '\*' to the directory name.

    // StringCchCopy(dir_size_tchar, MAX_PATH, wstr.c_str());
    // StringCchCat(dir_size_tchar, MAX_PATH, TEXT("\\*"));
    StringCchCopyA(dir_size_char, MAX_PATH, folder_start.c_str());
    StringCchCatA(dir_size_char, MAX_PATH, "\\*");

    // Find the first file in the directory.
    // hFind = FindFirstFile(dir_size_tchar, &LoFindData);
    hFind = FindFirstFileA(dir_size_char, &LoFindData);

    constexpr auto GetPathStr = [](const WIN32_FIND_DATAA& LoFindData) -> xstring 
    {
        xstring path_str;
        int count = 0;
        // while (LoFindData.cFileName[count] != '\0') {
        //     path_str += LoFindData.cFileName[count];
        //     ++count;
        // }
        while (LoFindData.cFileName[count] != '\0') {
            path_str += LoFindData.cFileName[count];
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
            //if (LoFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (LoFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                file_path_str = GetPathStr(LoFindData) + "\\";
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
                    tmp_path = GetPathStr(LoFindData);
                    if (*(folder_start.end() - 1) == '\\' || tmp_path[0] == '\\')
                        track_vec.push_back(folder_start + GetPathStr(LoFindData));
                    else
                        track_vec.push_back(folder_start + '\\' + GetPathStr(LoFindData));
                }
            }
            //} while (FindNextFile(hFind, &LoFindData) != 0);
        } while (FindNextFileA(hFind, &LoFindData) != 0);

        FindClose(hFind);
    }
    Rescue();
};
#elif defined(BxNix)
void RA::OS::Dir_Continued(const xstring scan_start, xvector<xstring>& track_vec, \
    const bool folders, const bool files, const bool recursive)
{
    Begin();
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
                    RA::OS::Dir_Continued(scan_start + "/" + dir_item, \
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
    Rescue();
}
#endif

// -------------------------------------------------------------------------------------------

RA::OS::OS()
{
    m_last_read = 'n';
};

RA::OS::~OS() {
    File.Close();
};

xvector<int> RA::OS::GetConsoleSize() // [columns, rows]
{
    Begin();
#if defined(BxWindows)

    CONSOLE_SCREEN_BUFFER_INFO screen_info;

    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &screen_info);
    return xvector<int>{
        static_cast<int>(1) + screen_info.srWindow.Right  - screen_info.srWindow.Left, // columns
        static_cast<int>(1) + screen_info.srWindow.Bottom - screen_info.srWindow.Top   // rows
    };
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return xvector<int>({ w.ws_col, w.ws_row });
#endif
    Rescue();
}

bool RA::OS::HasFileSyntax(const xstring& i_file) 
{
    Begin();
    if (!i_file.Match(R"(^([\./\\]+?)[\-\d\w\.\\/]+$)"))
        return false;
    else if (i_file.Scan(R"([^\\]\s)"))
        return false;
    else
        return true;
    Rescue();
}

bool RA::OS::AllHaveFileSyntax(const xvector<xstring>& i_files)
{
    Begin();
    for (const xstring& i_file : i_files) {
        if (!(RA::OS::HasFileSyntax(i_file)))
            return false;
    }return true;
    Rescue();
}


// ---------------------------------------------------------------------------------------------

void RA::OS::Touch(const xstring& new_file)
{
    Begin();
    if (new_file.size())
        File.m_name = RA::OS::FullPath(new_file);

    if (RA::OS::Has(File.m_name))
        return;

    File.SetWrite();
    Rescue();
}


void RA::OS::MKDIR(const xstring& folder)
{
    Begin();
    if (!folder.size()) return;

    xstring new_folder = RA::OS::FullPath(folder);

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
#if defined(BxWindows)
        if ((*iter)[1] == ':')
        {
            folder_path += *iter;
            continue;
        }
        folder_path += '\\' + *iter;
        CreateDirectoryA(folder_path.c_str(), NULL);
#elif defined(BxNix)
        folder_path += '/' + *iter;
        ::mkdir(folder_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
    }
    Rescue();
}


void RA::OS::CP(const xstring& old_location, const xstring& new_location)
{
    Begin();
    if (!(old_location.size() || new_location.size())) return;

    xstring old_path = RA::OS::FullPath(old_location);
    xstring new_path = RA::OS::FullPath(new_location);

    if (RA::OS::HasFile(old_path))
        RA::OS::CopyFile(old_path, new_path);
    else if (RA::OS::HasDir(new_path))
        RA::OS::CopyDir(old_path, new_path);
    else
        throw std::runtime_error("Location of Copy Not Found: " + old_location);
    Rescue();
}


void RA::OS::MV(const xstring& old_location, const xstring& new_location)
{
    Begin();
    xstring old_path = RA::OS::FullPath(old_location);
    xstring new_path = RA::OS::FullPath(new_location);

    if (!(old_location.size() || new_location.size())) return;

    if (RA::OS::HasFile(old_path))
        RA::OS::MoveFile(old_path, new_path);
    else if (RA::OS::HasDir(old_path))
        RA::OS::MoveDir(old_path, new_path);
    else
        throw std::runtime_error("Move Start Location Not Found: " + old_location);
    Rescue();
}


void RA::OS::RM(const xstring& del_file)
{
    Begin();
    if (!del_file.size()) return;

    xstring bad_file = RA::OS::FullPath(del_file);

    if (RA::OS::HasFile(bad_file))
        RA::OS::RemoveFile(bad_file);

    else if (RA::OS::HasDir(bad_file))
        RA::OS::RemoveDir(bad_file);
    Rescue();
}

// ---------------------------------------------------------------------------------------------

RA::OS RA::OS::Open(const xstring& new_file_name, const char write_method)
{
    Begin();
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
    Rescue();
}

RA::OS RA::OS::Close()
{
    Begin();
    File.Close();
    return *this;
    Rescue();
}

// os.Open(file_name).Read()
// os.RunConsoleCommand(command).Read()
xstring RA::OS::Read(char content, bool close_file /* = false */)
{
    Begin();
    content = (content == 'd') ? m_last_read : content;
    switch (content) {
    case 'f':  return this->InstRead();
    case 'c':  return CMD.m_out;
    default:  return xstring("none");
    }

    if (close_file)
        this->Close();
    Rescue();
}

xstring RA::OS::InstRead()
{
    Begin();
    if (File.m_handler != 'r')
        File.SetRead();

    File.m_data.clear();
    xstring line;
    errno = 0;
    if (File.m_in_stream.is_open()) {
        while (getline(File.m_in_stream, line))
            File.m_data += line + '\n';
    }
    else 
        ThrowIt("Error (" + RA::ToXString(errno) + "): File Locked: ", File.m_name);
    return File.m_data;
    Rescue();
}


xstring RA::OS::ReadFile(const xstring& FsFilename, bool FbRetry, bool FbUseBinaries)
{
    Begin();
    if (!FsFilename.size())
        return xstring();

    if (!HasFile(FsFilename))
        ThrowIt("File Does Not Exist: ", FsFilename);
    errno = 0;
    FILE* fp;
#ifdef BxWindows
    fopen_s(&fp, FsFilename.c_str(), "rb");
#else
    fp = fopen(file_name.c_str(), "rb");
#endif
    if (fp == nullptr && !FbRetry) {
        xstring err("Error (" + RA::ToXString(errno) + "): File Locked: ");
        err += FsFilename;
        throw std::runtime_error(err.c_str());
    }
    else if ((errno || fp == nullptr) && FbRetry)
    {
        if (fp) fclose(fp);
        return RA::OS::ReadStatMethod(FsFilename);
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
    rets.reserve(file_size);
    if (rbuffer && !FbUseBinaries)
    {
        int NonASCIICount = 0;
        for (auto i = 0; i < file_size; i++)
        {
            if (rbuffer[i] > 0 && rbuffer[i] < 128)
            {
                rets += rbuffer[i];
            }
            else if(rbuffer[i] > 128)
                NonASCIICount++;

            if (NonASCIICount > 10)
            {
                free(rbuffer);
                if (fp) fclose(fp);
                return xstring();
            }
        }
        free(rbuffer);
        if (fp) fclose(fp);
    }
    else if(rbuffer && FbUseBinaries)
    {
        for (auto i = 0; i < file_size; i++)
        {
            if (rbuffer[i] > 0 && rbuffer[i] < 128)
                rets += rbuffer[i];
        }
        free(rbuffer);
        if (fp) fclose(fp);
    }

    return rets;
    Rescue();
}

xstring RA::OS::ReadStatMethod(const xstring& file_name)
{
    Begin();
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
    Rescue();
}

xstring RA::OS::ReadStreamMethod(const xstring& file_name)
{
    Begin();
    if (!file_name.size())
        return xstring();

    std::ifstream os_file(file_name, std::ios::in | std::ios::binary); // iRA::OS::in for data intake 

    xstring file_data;
    xstring line;

    errno = 0;
    if (os_file.is_open()) {
        while (getline(os_file, line))
            file_data += line + '\n';
        os_file.close();
    }
    else {
        xstring err("Error (" + RA::ToXString(errno) + "): File Locked: ");
        err += file_name;
        throw std::runtime_error(err.c_str());
    }
    return file_data;
    Rescue();
}

xstring RA::OS::GetSTDIN()
{
    Begin();
    xstring LsLines = "";
#if defined(BxWindows)
    time_t startTime = time(NULL);
    RA::Atomic<bool> LbStarted = false;
    RA::Atomic<bool> LbHasData = false;
    RA::Atomic<bool> LbComplete = false;
    std::thread t1([&LsLines, &LbStarted, &LbHasData, &LbComplete]()
        {
            xstring LsLine;
            LbStarted = true;
            while (getline(std::cin, LsLine))
            {
                LbHasData = true;
                LsLines += LsLine + '\n';
            }
            LbComplete = true;
        });
    auto pid = t1.get_id();
    t1.detach();
    // check the InputReceived flag once every 50ms for 10 seconds

    while (!LbStarted)
        Nexus<>::Sleep(1);

    do {
        Nexus<>::Sleep(10);
    } while (time(NULL) < startTime + 1 && !LsLines.Size());

    if (!LbHasData)
        t1.~thread();
    else
        while (!LbComplete)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
#elif defined(BxNix)
ThrowIt("Not Yet Programmed");
#endif        
    return LsLines;
Rescue();
}

// you must use open before write
RA::OS RA::OS::Write(const xstring& content, bool store /* = false */)
{
    Begin();
    if (File.m_handler == 'a')
        File.SetAppend();
    else
        File.Clear();

    errno = 0;
    if (!File.m_out_stream.is_open())
        ThrowIt("Error (" + RA::ToXString(errno) + ") Unable to Open File: " + File.m_name + "\n");

    if (store)
        File.m_data = content;
    else
        File.m_data.clear();

    File.m_out_stream << content;

    m_last_read = 'f';
    return *this;
    Rescue();
}


xvector<xstring> RA::OS::Dir(const xstring& folder_start, const char mod1, const char mod2, const char mod3)
{
    Begin();
    // recursive = search foldres recursivly
    // folders   = return folders in search
    // files     = return files in search

    if (!folder_start.size()) return xvector<xstring>();

    xstring search_path = RA::OS::FullPath(folder_start);
    xvector<xstring> track_vec;

    if (folder_start == "" || (!RA::OS::HasDir(search_path))) {
        return track_vec;
    }

    const char* options = nullptr;
    options = new const char[4]{ mod1, mod2, mod3, '\0' };

    bool files = false;
    bool directories = false;
    bool recursive = false;

    constexpr auto set_mods = [](const char option, bool& files, bool& directories, bool& recursive) -> void {
        switch (option) {
        case 'f':files = true; break;
        case 'd':directories = true; break;
        case 'r':recursive = true;
        }
    };
    for (int i = 0; i < 3; i++)
        set_mods(options[i], files, directories, recursive);

    if (files == 0 && directories == 0) {
        return xvector<xstring>(1, xstring());
    }
    Dir_Continued(search_path, track_vec, directories, files, recursive);
    delete[] options;

    for (xstring& str : track_vec)
        str.RemoveNulls();

    return track_vec;
    Rescue();
}

RA::OS& RA::OS::RunConsoleCommand(const xstring& command, char leave)
{
    Begin();
    // leave styles
    // p = pass (nothing happens = defult)
    // t = throw exception if command fails
    // e = exit if command fails
#ifdef BxWindows
    CMD.m_cmd = command;
#else
    CMD.m_cmd = command + " 2>&1";
#endif

    CMD.m_out.clear();
    int buf_size = 512;
    char buffer[512];

#ifdef BxWindows
    FILE* file = _popen(CMD.m_cmd.c_str(), "r");
#else
    std::cout << cmd.m_cmd.c_str() << std::endl;
    FILE* file = popen(cmd.m_cmd.c_str(), "r");
#endif

    while (!feof(file)) {
        if (fgets(buffer, buf_size, file) != NULL)
            CMD.m_out += buffer;
    }

#ifdef BxWindows
    int returnCode = _pclose(file);
#else
    int returnCode = pclose(file);
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
    Rescue();
}

xstring RA::OS::operator()(const xstring& command, const char leave) {
    return this->RunConsoleCommand(command, leave).Read();
};

// ============================================================================================

void RA::OS::AssertFolderSyntax(const xstring& folder)
{
    Begin();
    if (!folder.Match(R"(^([\./\\]+?)[\-\d\w\.\\/]+$)")) {
        ThrowIt("Failed Dir Syntax = "
            R"(^([\./\\]+?)[\-\d\w\.\\/]+$)"
            "\n  what():  Dir Item: " + folder + "\n");
    }

    if (folder.Scan(R"([^\\]\s)")) {
        ThrowIt("You can't have a space in a dir item\n" \
            "  what():  without an escape char\n");
    }
    Rescue();
}

void RA::OS::MoveFile(const xstring& old_location, const xstring& new_location)
{
    Begin();
    if (!(old_location.size() || new_location.size())) return;

    xstring old_path = RA::OS::FullPath(old_location);
    xstring new_path = RA::OS::FullPath(new_location);
    try {
        std::ifstream  in(old_path, std::ios::in | std::ios::binary);
        std::ofstream out(new_path, std::ios::out | std::ios::binary);
        out << in.rdbuf();

        if (in.is_open()) in.close();
        if (out.is_open()) out.close();
    }
    catch (std::runtime_error & err) {
        ThrowIt(err.what());
    }

    RA::OS::RemoveFile(old_path);
    Rescue();
}

void RA::OS::MoveDir(const xstring& old_location, const xstring& new_location) 
{
    Begin();
    if (!(old_location.size() || new_location.size())) return;

    RA::OS::CopyDir(old_location, new_location);
    RA::OS::RemoveDir(old_location);
    Rescue();
}


void RA::OS::CopyFile(const xstring& old_location, const xstring& new_location)
{
    Begin();
    if (!(old_location.size() || new_location.size())) return;

    xstring old_path = RA::OS::FullPath(old_location);
    xstring new_path = RA::OS::FullPath(new_location);

    std::ifstream  in(old_path, std::ios::in | std::ios::binary);
    std::ofstream out(new_path, std::ios::out | std::ios::binary);

    out << in.rdbuf();
    if (in.is_open()) in.close();
    if (out.is_open()) out.close();
    Rescue();
}

void RA::OS::CopyDir(const xstring& old_location, const xstring& new_location)
{
    Begin();
    if (!(old_location.size() || new_location.size()))
        return;

    xstring old_path = RA::OS::FullPath(old_location);
    xstring new_path = RA::OS::FullPath(new_location);

    if (!RA::OS::HasDir(old_path))
        throw std::runtime_error("\nThis is not a folder\n" + old_path + '\n');


    if (RA::OS::HasFile(new_path))
        throw std::runtime_error("\nA file exists there\n" + new_path + '\n');

    xvector<xstring> old_folders = RA::OS::Dir(old_path, 'r', 'd');
    xvector<xstring> old_files = RA::OS::Dir(old_path, 'r', 'f');
    size_t old_size = old_path.size();

    xstring nested_dir_item;
    if (old_folders.size()) {
        for (xvector<xstring>::const_iterator it = old_folders.begin(); it < old_folders.end(); it++) {
            nested_dir_item = (*it).substr(old_size, (*it).size() - old_size);
            RA::OS::MKDIR(new_path + nested_dir_item);
        }
    }
    else {
        RA::OS::MKDIR(new_path);
    }

    if (old_files.size()) {
        for (xvector<xstring>::const_iterator it = old_files.begin(); it < old_files.end(); it++) {
            nested_dir_item = (*it).substr(old_size, (*it).size() - old_size);
            RA::OS::CopyFile(*it, new_path + nested_dir_item);
        }
    }
    Rescue();
}

void RA::OS::RemoveFile(const xstring& item)
{
    Begin();
    xstring fitem = RA::OS::FullPath(item);

    if (!OS_O::Dir_Type::HasFile(fitem))
        throw std::runtime_error(std::string("Error Filename: ") + fitem + " Does Not Exist!\n");

    errno = 0;
    try {
#if defined(BxWindows)
        DeleteFileA(fitem.c_str());
#elif defined(BxNix)
        remove(fitem.c_str());
#endif

        if (errno)
            throw;
    }
    catch (...) {
        xstring err = "Failed (" + RA::ToXString(errno) + "): Failed to delete file: '" + fitem + "'\n";
        throw std::runtime_error(err);
    }
    Rescue();
}


void RA::OS::RemoveDir(const xstring& folder)
{
    Begin();
    if (!folder.size()) return;

    xstring del_path = RA::OS::FullPath(folder);

    xvector<xstring> dir_items = RA::OS::Dir(del_path, 'r', 'f', 'd');
    dir_items.push_back(del_path);

    std::multimap<size_t, xstring> dir_size_mp; // I only have a unordered_map of the extended stl at this time

    for (xvector<xstring>::const_iterator it = dir_items.begin(); it != dir_items.end(); it++)
        dir_size_mp.insert({ it->Count("([\\\\/][^\\\\/\\s])"), *it });

    std::multimap<size_t, xstring>::const_reverse_iterator dir_item;

    constexpr auto delete_dir_item = [](const xstring& dir_item) -> void
    {
    #if defined(BxWindows)
        if (RA::OS::HasDir(dir_item))
            RemoveDirectoryA(dir_item.c_str());
        else if (RA::OS::HasFile(dir_item))
            DeleteFileA(dir_item.c_str());
    #elif defined(BxNix)
        if (RA::OS::HasDir(dir_item))
            ::rmdir(dir_item.c_str());
        else if (RA::OS::HasFile(dir_item))
            remove(dir_item.c_str());
    #endif
    };

    for (dir_item = dir_size_mp.rbegin(); dir_item != dir_size_mp.rend(); dir_item++)
        delete_dir_item(dir_item->second);
    Rescue();
}

void RA::OS::ClearFile(const xstring& i_file)
{
    Begin();
    if (!i_file.size())
        return;

    xstring file = RA::OS::FullPath(i_file);
    std::ofstream ofs;
    ofs.open(file.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (ofs.is_open())
        ofs.close();
    Rescue();
}

// <<<< file managment ------------------------------------------------------------------------
// ============================================================================================

