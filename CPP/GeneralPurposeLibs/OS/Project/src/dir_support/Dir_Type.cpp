#include "dir_support/Dir_Type.h"


#include<iostream>
#if defined(WIN_BASE)
#include<Windows.h>
#include<direct.h>  
#elif defined(NIX_BASE)
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

#if defined(NIX_BASE)

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

#elif defined(WIN_BASE)

    struct stat st;
    if (stat(input.Sub("\\\\", "/").c_str(), &st) == 0) {
        if (st.st_mode & S_IFDIR) {
            return DT::directory;
        } else if (st.st_mode & S_IFREG) {
            return DT::file;
        } else {
            return DT::none;
        }
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
#if defined(NIX_BASE)
    char result[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", result, FILENAME_MAX);
    bwd = xstring(result).Sub("/[^/]*$", "");

#elif defined(WIN_BASE)
    char buf[256];
    GetCurrentDirectoryA(256, buf);
    bwd = buf;
#endif
    bwd.RemoveNulls();
    return bwd;
}

xstring RA::OS_O::Dir_Type::PWD() {
    xstring pwd;
#if defined(NIX_BASE)
    char c_pwd[256];
    if (NULL == getcwd(c_pwd, sizeof(c_pwd))) {
        perror("can't get current dir\n");
        throw;
    }
    pwd = c_pwd;

#elif defined(WIN_BASE)
    char* buffer; 
    if ((buffer = _getcwd(NULL, 0)) == NULL) {
        perror("can't get current dir\n"); throw;
    } else{
        pwd = buffer;
        free(buffer);
    }
#endif
    pwd.RemoveNulls();
    return pwd;
}


xstring RA::OS_O::Dir_Type::Home() {
    xstring home_str;
#if defined(NIX_BASE)
    struct passwd *pw = getpwuid(getuid());
    home_str = pw->pw_dir;

#elif defined(WIN_BASE)
    char* path_str;
    size_t len;
    _dupenv_s( &path_str, &len, "USERPROFILE" );
    home_str = path_str;
    free(path_str);
    // _dupenv_s( &path_str, &len, "pathext" ); TODO ADD THIS
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
#if defined(WIN_BASE)
        return PWD() + file(1).Sub(s_forwardslash, "\\\\").RightTrim("\\/");
#elif defined(NIX_BASE)
        return PWD() + file(1).Sub(s_backslash, "/").RightTrim("\\/");
#endif
    }


#ifdef WIN_BASE
    char full[_MAX_PATH];
#else
    char full[PATH_MAX];
#endif
    xstring full_path;
#if defined(WIN_BASE)
    const char* unused = _fullpath(full, file.Sub(s_forwardslash, "\\\\").c_str(), _MAX_PATH);
#elif defined(NIX_BASE)
    xstring awkward_path = PWD() + '/' + file;
    const char* unused = realpath(awkward_path.Sub(s_forwardslash, "/").c_str(), full);
#endif
    full_path = full;
    full_path.RemoveNulls();
    return full_path.RightTrim("\\/");
}
