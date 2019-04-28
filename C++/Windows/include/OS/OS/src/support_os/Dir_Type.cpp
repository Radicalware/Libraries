#include "../../include/support_os/Dir_Type.h"

#include<iostream>
#include "re.h"

#if defined(WIN_BASE)
#include<Windows.h>
#include<direct.h>  
#elif defined(NIX_BASE)
#include<dirent.h>
#include<sys/stat.h>
#include<unistd.h>
#include<pwd.h>
#endif


Dir_Type::Dir_Type() {}
Dir_Type::~Dir_Type() {}


Dir_Type::dir_type Dir_Type::has(const std::string& input) {

#if defined(NIX_BASE)
    std::string file = re::sub("\\\\", "/", input);

    struct stat path_st;
    char dt = 'n';

    stat(file.c_str(), &path_st);
    if(S_ISREG(path_st.st_mode) ) {
        dt = 'f' ;
    }

    if(S_ISDIR(path_st.st_mode)){
        dt = 'd';
    }

    switch(dt){
        case 'f': return os_file;break;
        case 'd': return os_directory;break;
        case 'n': return os_none;
    }    

#elif defined(WIN_BASE)
    std::string file = re::sub("/", "\\\\", input);

    struct stat st;
    if (stat(file.c_str(), &st) == 0) {
        if (st.st_mode & S_IFDIR) {
            return os_directory;
        } else if (st.st_mode & S_IFREG) {
            return os_file;
        } else {
            return os_none;
        }
    }
#endif
    return os_none;
}

bool Dir_Type::file(const std::string& file) {
    return (this->has(file) == os_file);
}

bool Dir_Type::directory(const std::string& folder) {
    return (this->has(folder) == os_directory);
}


std::string Dir_Type::dir_item_str(const std::string& input) {

    dir_type dt = os_none;
    dt = this->has(input);

    switch(dt){
        case os_file:      return "file";break;
        case os_directory: return "directory";break;
        case os_none:      return "none";
        default:           return "none";
    }
}


std::string Dir_Type::bpwd() {
#if defined(NIX_BASE)
    char result[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", result, FILENAME_MAX);
    return re::sub("/[^/]*$", "", std::string(result, (count > 0) ? count : 0));

#elif defined(WIN_BASE)
    char buf[256];
    GetCurrentDirectoryA(256, buf);
    return std::string(buf);
#endif
}

std::string Dir_Type::pwd() {
#if defined(NIX_BASE)
    char c_pwd[256];
    if (NULL == getcwd(c_pwd, sizeof(c_pwd))) {
        perror("can't get current dir\n");
        throw;
    }
    return std::string(c_pwd);

#elif defined(WIN_BASE)
    char* buffer; 
    std::string pwd;
    if ((buffer = _getcwd(NULL, 0)) == NULL) {
        perror("can't get current dir\n"); throw;
    } else{
        pwd = buffer;
        free(buffer);
    }
    return pwd;
#endif
}


std::string Dir_Type::home() {
#if defined(NIX_BASE)
    struct passwd *pw = getpwuid(getuid());
    const char *char_home_dir = pw->pw_dir;
    return std::string(char_home_dir);

#elif defined(WIN_BASE)
    return std::string(getenv("USERPROFILE"));
#endif
}