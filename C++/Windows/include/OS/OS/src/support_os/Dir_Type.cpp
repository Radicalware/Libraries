#include "../../include/support_os/Dir_Type.h"

#include<iostream>
#include "re.h"

#ifdef NIX_BASE
#include<dirent.h>
#include<sys/stat.h>
#endif


Dir_Type::Dir_Type() {}
Dir_Type::~Dir_Type() {}


Dir_Type::dir_type Dir_Type::has(const std::string& input) {

#if defined(NIX_BASE)
    std::string file = re::sub("\\\\", "/", input);

    struct stat path_st;
    char dt = 'n';

    stat(file.c_str(), &path_st);
    if(S_ISREG(path_st.st_mode)){
        dt = 'f' ;
    }

    if(S_ISDIR(path_st.st_mode) && (path_st.st_mode & S_IFDIR != 0)){
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
    } else {
        return os_none;
    }
#endif
}

bool Dir_Type::file(const std::string& file) {
    return (this->has(file) == os_file);
}

bool Dir_Type::directory(const std::string& folder) {
    return (this->has(folder) == os_directory);
}


std::string Dir_Type::dir_item_type(const std::string& input) {

#if defined(NIX_BASE)
    std::string file = re::sub("\\\\", "/", input);

    struct stat path_st;
    char dt = 'n';
    
    stat(file.c_str(), &path_st);
    if(S_ISREG(path_st.st_mode)){
        dt = 'f' ;
    }

    if(S_ISDIR(path_st.st_mode) && (path_st.st_mode & S_IFDIR != 0)){
        dt = 'd';
    }

    switch(dt){
        case 'f': return "file";break;
        case 'd': return "directory";break;
        case 'n': return "none";
    }


#elif defined(WIN_BASE)
    std::string file = re::sub("/", "\\\\", input);

    struct stat st;
    if (stat(file.c_str(), &st) == 0) {
        if (st.st_mode & S_IFDIR) {
            return "os_directory";
        } else if (st.st_mode & S_IFREG) {
            return "os_file";
        } else {
            return "os_none";
        }
    } else {
        return "os_none";
    }
#endif
}
