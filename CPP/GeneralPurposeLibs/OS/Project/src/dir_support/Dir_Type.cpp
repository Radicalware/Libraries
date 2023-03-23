#include "dir_support/Dir_Type.h"


#include<iostream>
#ifdef BxWindows
#include<Windows.h>
#include<direct.h>  
#else
#include<dirent.h>
#include<sys/stat.h>
#include<unistd.h>
#include<pwd.h>
#endif


const unsigned char RA::OS_O::Dir_Type::IsFile   = 0x8;
const unsigned char RA::OS_O::Dir_Type::IsFolder = 0x4;

RE2 RA::OS_O::Dir_Type::s_backslash(R"((\\\\))");
RE2 RA::OS_O::Dir_Type::s_back_n_forward_slashes(R"([\\\\/]+)");
RE2 RA::OS_O::Dir_Type::s_forwardslash(R"(/)");

RA::OS_O::Dir_Type::Dir_Type() {}
RA::OS_O::Dir_Type::~Dir_Type() {}


RA::OS_O::Dir_Type::DT RA::OS_O::Dir_Type::GetDirType(const xstring& input) {

#if defined(BxWindows)

    struct stat st;
    if (stat(input.Sub("\\\\", "/").c_str(), &st) == 0) {
        if (st.st_mode & S_IFDIR) {
            return DT::directory;
        }
        else if (st.st_mode & S_IFREG) {
            return DT::file;
        }
        else {
            return DT::none;
        }
    }
#elif defined(BxNix)
    struct stat path_st;
    char dt = 'n';

    stat(input.Sub("\\\\", "/").c_str(), &path_st);
    if(S_ISREG(path_st.st_mode) ) {
        dt = 'f' ;
    }

    if(S_ISDIR(path_st.st_mode)){
        dt = 'd';
    }

    switch(dt){
        case 'f': return DT::file;break;
        case 'd': return DT::directory;break;
        case 'n': return DT::none;
    } 
#endif
    return DT::none;
}

bool RA::OS_O::Dir_Type::Has(const xstring& item)
{
    return (Dir_Type::HasFile(item) || Dir_Type::HasDir(item));
}

bool RA::OS_O::Dir_Type::HasFile(const xstring& file) {
    return (Dir_Type::GetDirType(file) == DT::file);
}

bool RA::OS_O::Dir_Type::HasDir(const xstring& folder) {
    return (Dir_Type::GetDirType(folder) == DT::directory);
}

bool RA::OS_O::Dir_Type::HasFile(xstring&& file) {
    return (Dir_Type::GetDirType(file) == DT::file);
}

bool RA::OS_O::Dir_Type::HasDir(xstring&& folder) {
    return (Dir_Type::GetDirType(folder) == DT::directory);
}

xstring RA::OS_O::Dir_Type::GetDirItem(const xstring& input) {

    DT dt = DT::none;
    dt = Dir_Type::GetDirType(input);

    switch(dt){
        case DT::file:      return "file";break;
        case DT::directory: return "directory";break;
        case DT::none:      return "none";
        default:            return "none";
    }
}


xstring RA::OS_O::Dir_Type::BWD() {
    xstring bwd;

#if defined(BxWindows)
    char buf[256];
    GetCurrentDirectoryA(256, buf);
    bwd = buf;
#elif defined(BxNix)
    char result[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", result, FILENAME_MAX);
    bwd = xstring(result).Sub("/[^/]*$", "");
#endif
    bwd.RemoveNulls();
    return bwd;
}

xstring RA::OS_O::Dir_Type::PWD() {
    xstring pwd;
#if defined(BxWindows)
    char* buffer;
    if ((buffer = _getcwd(NULL, 0)) == NULL) {
        perror("can't get current dir\n"); throw;
    }
    else {
        pwd = buffer;
        free(buffer);
    }
#elif defined(BxNix)
    char c_pwd[256];
    if (NULL == getcwd(c_pwd, sizeof(c_pwd))) {
        perror("can't get current dir\n");
        throw;
}
    pwd = c_pwd;
#endif
    pwd.RemoveNulls();
    return pwd;
}


xstring RA::OS_O::Dir_Type::Home() {
    xstring home_str;
#if defined(BxWindows)
    char* path_str;
    size_t len;
    _dupenv_s(&path_str, &len, "USERPROFILE");
    home_str = path_str;
    free(path_str);
#elif defined(BxNix)
    struct passwd* pw = getpwuid(getuid());
    home_str = pw->pw_dir;
#endif
    home_str.RemoveNulls();
    return home_str;
}

xstring RA::OS_O::Dir_Type::FullPath(const xstring& file)
{
    if (file[0] != '.') // no path traversal
        return file;

    if (!file.size()) // no input
        return PWD();
        
    if(file[0] == '.' && file.size() == 1) // present working directory
        return PWD();

    else if (file[0] == '.' && (file[1] == '/' || file[1] == '\\')) // continued directory
    {
#if defined(BxWindows)
        return PWD() + file(1).InSub('/', '\\').RightTrim("\\/");
#elif defined(BxNix)
        return PWD() + file(1).InSub('\\', '//').RightTrim("\\/");
#endif
    }

#if defined(BxWindows)
    char full[_MAX_PATH];
#elif defined(BxNix)
    char full[PATH_MAX];
#endif

    xstring full_path;
#if defined(BxWindows)
    const char* unused = _fullpath(full, file.Sub('/', '\\').c_str(), _MAX_PATH);
#elif defined(BxNix)
    xstring awkward_path = PWD() + '/' + file;
    const char* unused = realpath(awkward_path.c_str(), full);
#endif
    full_path = full;
    full_path.RemoveNulls();
    return full_path.RightTrim("\\/");
}
