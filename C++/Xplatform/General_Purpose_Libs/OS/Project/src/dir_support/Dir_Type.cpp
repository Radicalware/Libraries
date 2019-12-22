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


const unsigned char OS_O::Dir_Type::IsFile   = 0x8;
const unsigned char OS_O::Dir_Type::IsFolder = 0x4;

OS_O::Dir_Type::Dir_Type() {}
OS_O::Dir_Type::~Dir_Type() {}


OS_O::Dir_Type::DT OS_O::Dir_Type::Get_Dir_Type(const xstring& input) {

#if defined(NIX_BASE)

    struct stat path_st;
    char dt = 'n';

    stat(input.sub("\\\\", "/").c_str(), &path_st);
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
    if (stat(input.sub("\\\\", "/").c_str(), &st) == 0) {
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

bool OS_O::Dir_Type::Has(const xstring& item)
{
    return (Dir_Type::Has_File(item) || Dir_Type::Has_Dir(item));
}

bool OS_O::Dir_Type::Has_File(const xstring& file) {
    return (Dir_Type::Get_Dir_Type(file) == DT::file);
}

bool OS_O::Dir_Type::Has_Dir(const xstring& folder) {
    return (Dir_Type::Get_Dir_Type(folder) == DT::directory);
}

bool OS_O::Dir_Type::Has_File(xstring&& file) {
    return (Dir_Type::Get_Dir_Type(file) == DT::file);
}

bool OS_O::Dir_Type::Has_Dir(xstring&& folder) {
    return (Dir_Type::Get_Dir_Type(folder) == DT::directory);
}


xstring OS_O::Dir_Type::Dir_Item_Str(const xstring& input) {

    DT dt = DT::none;
    dt = Dir_Type::Get_Dir_Type(input);

    switch(dt){
        case DT::file:      return "file";break;
        case DT::directory: return "directory";break;
        case DT::none:      return "none";
        default:            return "none";
    }
}


xstring OS_O::Dir_Type::BWD() {
#if defined(NIX_BASE)
    char result[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", result, FILENAME_MAX);
    return xstring(result).sub("/[^/]*$", "");

#elif defined(WIN_BASE)
    char buf[256];
    GetCurrentDirectoryA(256, buf);
    return xstring(buf);
#endif
}

xstring OS_O::Dir_Type::PWD() {
#if defined(NIX_BASE)
    char c_pwd[256];
    if (NULL == getcwd(c_pwd, sizeof(c_pwd))) {
        perror("can't get current dir\n");
        throw;
    }
    return xstring(c_pwd);

#elif defined(WIN_BASE)
    char* buffer; 
    xstring pwd;
    if ((buffer = _getcwd(NULL, 0)) == NULL) {
        perror("can't get current dir\n"); throw;
    } else{
        pwd = buffer;
        free(buffer);
    }
    return pwd;
#endif
}


xstring OS_O::Dir_Type::Home() {
#if defined(NIX_BASE)
    struct passwd *pw = getpwuid(getuid());
    const char *char_home_dir = pw->pw_dir;
    return xstring(char_home_dir);

#elif defined(WIN_BASE)
    char* path_str;
    size_t len;
    _dupenv_s( &path_str, &len, "USERPROFILE" );
    xstring ret_str = path_str;
    free(path_str);
    return ret_str;
    // _dupenv_s( &path_str, &len, "pathext" ); TODO ADD THIS
#endif
}
