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
                // This is the only non-std lib required for OS.h


#include<iostream>
#include<vector>
#include<string>
#include<stdexcept>
#include<stdio.h>      // defines FILENAME_MAX or PATH_MAX

#include<fstream>      // file-handling

#include<dirent.h>     // read/write

#include<stdio.h>      // popen
#include<sys/stat.h>   // mkdir 
#include<unordered_map>

#include<assert.h>
#include<stdio.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;

OS os;

OS::OS(int c_argc, char** c_argv):
    m_argc(c_argc), m_argv(c_argv){
        (this)->set_args(m_argc, m_argv);
    };


OS::OS(){
    std::string tmp = "xnone";
};
OS::~OS(){};
// ------------------------------------------
OS OS::open(std::string new_file_name, char write_method){
    // a = append     (append then writes like in python)
    // w = write mode (clears then writes like in python)

    m_write_method = write_method;
    m_file_name = new_file_name;
    switch(m_write_method) {
        case 'a' :  m_write_method = 'a';
                    break;
        case 'w' :  m_write_method = 'w';
                    this -> clear_file(m_file_name);
    }   
    m_last_read = 'f';
    return *this;
}

// os.open(file_name).read()
// os.popen(command).read()
std::string OS::read(char content){
    content = (content == 'n') ? m_last_read : content;

    switch(content) {
        case 'f' :  return this -> read_file(); break;
        case 'c' :  return m_command; break;
        default  :  return "none"; 
    }    
}

std::string OS::read_file(){
    std::ifstream os_file(m_file_name);
    std::string line; 
    m_file_data.clear();

    if(os_file.is_open()){
        while (getline(os_file, line)){
            m_file_data += line + '\n';
        }
        os_file.close();
    }

    return m_file_data;
}

OS OS::write(std::string content, char write_method){

    m_write_method = (write_method == 'n') ? m_write_method : write_method;

    switch(m_write_method) {
        case 'a' :  m_write_method = 'a';
                    m_file_data = this -> read() + content;
                    break;
        
        case 'w' :  m_write_method = 'w';
                    m_file_data = content;
                    this -> clear_file(m_file_name);
    }   

    const char* content_ptr = &m_file_data[0];

    std::ofstream os_file(m_file_name);
    if (os_file.is_open()){
        os_file << content_ptr;
        os_file.close();
    }
    m_last_read = 'f';
    return *this;
}

// ------------------------------------------

void OS::dir_continued(string scan_start, vector<string>& vec_track, bool folders, bool files, bool recursive, bool star){

    DIR *current_dir = opendir (scan_start.c_str()); // starting dir given as a string
    // "dir_item" can be anything in the "current_dir" such as a new folder, file, binary, etc.
    while (struct dirent *dir_item_ptr = readdir(current_dir)){
        string dir_item =  (dir_item_ptr->d_name); // structure points to the getter to retrive the dir_item's name.
        if (dir_item != "." and dir_item != "./" and dir_item != ".."){
            if (dir_item_ptr->d_type == DT_DIR){
                if(folders){
                    std::string prep = (star) ? "*": "";
                    vec_track.push_back(prep + scan_start + "/" + dir_item);
                }
                if(recursive){
                    this->dir_continued(scan_start + "/" + dir_item, vec_track, folders, files, recursive, star); 
                    // recursive function
                }
            }else if(dir_item == "read"){
                break; // full dir already read, leave the loop
            }else if(files){
                vec_track.push_back(scan_start + "/" + dir_item);
            }
        }
    }
    closedir (current_dir); 
}

vector<string> OS::dir(string folder_start, string mod1, string mod2, string mod3, string mod4){
    // dir(folder_to_start_search_from, &files_to_return,'r','f');
    // recursive = search foldres recursivly
    // folders   = return folders in search
    // files     = return files in search
    // star      = place a '*' in front of folders

    this->assert_folder_syntax(folder_start);

    vector<string> options = {mod1, mod2, mod3, mod4};
    bool folders = 0;
    bool files = 0;
    bool recursive = 0;
    bool star = 0;
    std::vector<string>::iterator iter;
    for (iter = options.begin(); iter != options.end(); ++iter){
        if (*iter == "folders"){
            folders = true;
        }else if(*iter == "files"){
            files = true;
        }else if(*iter == "recursive" || *iter == "recurse"){
            recursive = true;
        }else if(*iter == "star")
            star = true;
    }
    if (files == 0 && folders == 0){
        return vector<string>({""});
    }
    vector<string> vec_track;
    dir_continued(folder_start, vec_track, folders, files, recursive, star);
    return vec_track;
}



#ifndef MSWINDOWS
    std::string OS::pwd(){
        char result[ FILENAME_MAX ];
        ssize_t count = readlink( "/proc/self/exe", result, FILENAME_MAX );
        return re::sub("/[^/]*$","",std::string( result, (count > 0) ? count : 0 ));
    }
#else
    std::string OS::pwd(){
        char result[ FILENAME_MAX ];
        return std::string( result, GetModuleFileName( NULL, result, FILENAME_MAX ));
    }
#endif

// Replace popen and pclose with _popen and _pclose for Windows.
OS OS::popen(const std::string command, char leave){
    // leave styles
    // p = pass (nothing happens = defult)
    // t = throw exception if command fails
    // e = exit if command fails
    const char* commmand_ptr = &command[0];
    m_command.clear();
    int buf_size = 512;
    char buffer[buf_size];
    FILE* file = ::popen(commmand_ptr, "r");
    while (!feof(file)) {
        if (fgets(buffer, buf_size, file) != NULL)
            m_command += buffer;  
    }     
    int returnCode = pclose(file);
    if(returnCode){
        cout << "\nConsole Command Failed:\n" << command << endl;
        
        switch(leave){
            case 't': throw;
            case 'e': exit(1);
            // no need for break
        }
    }
    m_last_read = 'c';
    return *this;
}

// ============================================================================================

bool OS::findFile(std::string file){ // no '_' based on ord namespace syntax with keyword 'find'

    assert_folder_syntax(file);
    std::ifstream os_file(&file[0]);
    bool file_exists = false;
    if (os_file.is_open()){
        file_exists = true;
        os_file.close();
    }
    return file_exists;
}

OS OS::move_file(std::string old_location, std::string new_location){
    if (!new_location.size()){
        new_location = old_location;
        old_location = m_file_name;
    }
    this->assert_folder_syntax(old_location, new_location);

    std::ifstream  in(old_location,  std::ios::in | std::ios::binary);
    std::ofstream out(new_location, std::ios::out | std::ios::binary);
    out << in.rdbuf();
    ::remove(&old_location[0]);
    return *this;
}

OS OS::copy_file(std::string old_location, std::string new_location){
    if (!new_location.size()){
        new_location = old_location;
        old_location = m_file_name;
    }
    this->assert_folder_syntax(old_location, new_location);
    std::ifstream  in(old_location,  std::ios::in | std::ios::binary);
    std::ofstream out(new_location, std::ios::out | std::ios::binary);
    out << in.rdbuf();
    return *this;
}

OS OS::clear_file(std::string content){
    std::string file_to_wipe = (content == "") ? m_file_name : content;
    this->assert_folder_syntax(file_to_wipe);
    std::ofstream ofs;
    ofs.open(file_to_wipe, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    return *this;
}


OS OS::delete_file(std::string content){
    std::string file_to_delete = (content == "") ? m_file_name : content;
    this->assert_folder_syntax(file_to_delete);
    ::remove( &file_to_delete[0] );
    return *this;
}

OS OS::mkdir(std::string folder){
    this->assert_folder_syntax(folder);
    char seperator = (PLATFORM == "nix") ? '/' : '\\';
    std::string folder_path = re::sub("[a-zA-Z0-9_].*$","",folder);
    std::vector<std::string> folders = re::split("[\\\\|/](?=[^\\s])", re::sub("^[\\.\\|/]*|[\\|/]*$","",folder));
    std::vector<std::string>::iterator iter;
    for (iter = folders.begin(); iter != folders.end(); ++iter){
        if (re::match("[a-zA-Z0-9_]*",*iter)){
            folder_path += seperator + *iter;
            if(this->findFile(folder_path) == false){
                ::mkdir(&folder_path[0], S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            }
        }
    }
    return *this;
}

OS OS::rmdir(std::string folder){
    this->assert_folder_syntax(folder);

    std::vector<std::string> filesystem = dir(folder,"files","folders","recurse","star");

    int max_slash = -1;
    auto get_max_path = [&filesystem, &max_slash]() -> int {
        std::for_each(filesystem.begin(), filesystem.end(), 
            [&max_slash](std::string item) -> int {
                int current_count = re::count("[^\\\\|/][\\\\|//]", item);
                if (max_slash < current_count){
                    max_slash = current_count;
                };
            }
        );
    };

    auto rm_slash_count = [&filesystem, &max_slash]() {
        int count = 0;
        std::for_each(filesystem.begin(), filesystem.end(),
            [&](std::string item) {
                if(re::count("[^\\\\|/][\\\\|//]", item) == max_slash && item.size()){
                    if(item[0] == '*'){
                        ::rmdir(&(item.substr(1,item.size()-1))[0]);
                    }else{
                        ::remove( &item[0]);
                    }
                    filesystem[count] = "";
                };
                count += 1;
            }
        );
    };

    get_max_path(); 
    rm_slash_count();
    while(max_slash > 1){  
        max_slash = 0;  
        get_max_path();
        rm_slash_count();       
    }
    ::rmdir(&folder[0]);
   
    return *this;
}

void OS::assert_folder_syntax(std::string folder1, std::string folder2){
    assert(re::match("^[\\./\\\a-zA-Z0-9_]*$",folder1));
    assert(!re::scan("[^\\\\]\\s",folder1));
    if(folder2.size()){
        assert(re::match("^[\\./\\\a-zA-Z0-9_]*$",folder2));
        assert(!re::scan("[^\\\\]\\s",folder2));
    }
}
// <<<< file managment
// ============================================================================================
// >>>> args

OS OS::set_args(int argc, char** argv){
    m_all_args.resize(argc);
    bool sub_arg = false;
    bool first_header = false;
    int first_sub = 0;
    std::vector<std::string> prep_sub_arg;
    std::string current_base;


    for(int arg = 0; arg <= argc; arg++){ // loop through argv
        if(arg != argc)
            m_all_args[arg] = argv[arg];
        if(first_header){
            if(arg == argc && prep_sub_arg.size() > 0){
                // last call to append prep_sub_arg to KVP
                if(m_args.count(current_base)){
                    for(std::string str: prep_sub_arg)
                        m_args.at(current_base).push_back(str);
                }else{
                    // std::unordered_map<std::string, std::vector<std::string> > m_args; // {base_arg, [sub_args]}
                    m_args.insert(std::make_pair(current_base, prep_sub_arg));
                }
                for(std::string str: prep_sub_arg)m_sub_args_str += str + ' ';
            }else if(arg < argc){
                if(argv[arg][0] == '-' || argv[arg][0] == '/'){
                    if (prep_sub_arg.size() > 0){
                        m_sub_args.push_back(prep_sub_arg);
                        if(m_args.count(current_base)){
                            for(std::string str: prep_sub_arg)
                                m_args.at(current_base).push_back(str);
                        }else{
                            m_args.insert(std::make_pair(current_base, prep_sub_arg));
                        }
                        for(std::string str: prep_sub_arg)m_sub_args_str += str + ' ';
                        prep_sub_arg.clear();
                        current_base = m_all_args[arg];
                        m_bargs.push_back(current_base);
                    }else if(!findKey(m_all_args[arg])){ // if the key doesn't already exist
                        m_args.insert(std::make_pair(m_all_args[arg], std::vector<std::string>{""}));
                    }
                }else{
                    prep_sub_arg.push_back(m_all_args[arg]);
                }
            }
        }else if(argv[arg][0] == '-' or argv[arg][0] == '/'){
            first_header = true;
            m_bargs.push_back(m_all_args[arg]);
            current_base = m_all_args[arg];
        }
    }

    for(std::string& str: m_all_args) m_all_args_str += str + ' ';
    for(std::string& str: m_bargs) m_bargs_str += str + ' ';

    return *this;   
}

// -------------------------------------------------------------------------------
std::string OS::file_data(){return m_file_data;}
std::string OS::file_name(){return m_file_name;}

std::string OS::command()  {return m_command;}
std::string OS::read_command()  {return m_command;}
std::string OS::cmd()  {return m_command;}
// -------------------------------------------------------------------------------
// All arg usage which which should only be used for small C++ tools.

std::vector<std::string> OS::argv(){return m_all_args;}
int OS::argc(){return m_all_args.size();}

std::string OS::operator[](int value){
    if (m_all_args.size() <= value){
        return "";
    }else{
        return m_all_args[value];
    }
}
std::unordered_map<std::string, std::vector<std::string> > OS::args(){return m_args;}

// -------------------------------------------------------------------------------

std::string OS::argv_str(){return m_all_args_str;}

bool OS::findArg(std::string find_arg){
    for(auto&arg : m_all_args){
        if (arg == find_arg)
            return true;
    }return false;
}


std::vector<std::string> OS::keys(){return m_bargs;}
std::string OS::keys_str(){return m_bargs_str;}

std::vector< std::vector<std::string> > OS::keyValues(){return m_sub_args;}
std::string OS::keyValues_str(){return m_sub_args_str;}

// -------------------------------------------------------------------------------
// These are the arg functions you will mostly use
// 1st you will identify if the key exist with "findKey()"
// 2nd you will either return it's values with "keyValues()"
//     or you will get the bool for the existence of its value
//     "findKeyValue()" for control flow of the program




std::string OS::keyValue(std::string key, int i){return m_args.at(key)[i];}

std::vector<std::string> OS::keyValues(std::string key){return m_args.at(key);}

std::vector<std::string> OS::operator[](std::string key){return m_args.at(key);}

bool OS::findKey(std::string key){
    std::vector<string>::iterator iter;
    for (iter = m_bargs.begin(); iter != m_bargs.end(); ++iter){
        if(*iter == key){
            return true;
        }
    }return false;
};
bool OS::findKeyValue(const std::string& key,const std::string& value){
    for(std::vector<std::string>::const_iterator iter = m_args.at(key).begin(); \
            iter != m_args.at(key).end(); iter++){
        if (*iter == value){
            return true;
        }
    }return false;
}

bool OS::operator()(const std::string& key, const std::string& value){
    for(std::vector<std::string>::const_iterator iter = m_args.at(key).begin(); \
            iter != m_args.at(key).end(); iter++){
        if (*iter == value){
            return true;
        }
    }return false;
}

// -------------------------------------------------------------------------------
template<class T = std::string>
void OS::p(T str){cout << endl << "------\n" << str << endl;}
void OS::d(int i){cout << endl << "---{dbg: " << i << "}---" << endl;}
// -------------------------------------------------------------------------------