#ifndef _OS_H_
#define _OS_H_

#include<iostream>
#include<vector>
#include<string>
#include<stdexcept>

#include<fstream>    // dir

#include<dirent.h>   // read/write
#include<string>

#include<stdio.h>    // popen

#include<unordered_map>


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

using std::cout;
using std::endl;
using std::string;
using std::vector;


class OS
{
private:
    std::string m_command;
    std::string m_file_name;
    std::string m_file_data = "  ";

    char m_last_read = 'n';
    char m_write_method = 'a';

    int    m_argc;
    char** m_argv;
    
    std::unordered_map<std::string, std::vector<std::string> > m_args; // {base_arg, [sub_args]}

    std::vector<std::string> m_all_args; 
    std::string m_all_args_str;

    std::vector<std::string> m_bargs; // base args
    std::string m_bargs_str;
    std::vector< std::vector<std::string> > m_sub_args; // sub args
    std::string m_sub_args_str;

    std::string blank {""};

    std::vector<std::string> blank_vec;

public:

    OS(int c_argc, char** c_argv):
    m_argc(c_argc), m_argv(c_argv)
    {(this)->set_args(m_argc, m_argv);};


    OS(){
        std::string tmp = "xnone";
        blank_vec.push_back(tmp);
    };
    ~OS(){};
    // ------------------------------------------
    OS open(std::string new_file_name, char write_method = 'a'){
        // a = append     (append then writes like in python)
        // w = write mode (clears then writes like in python)

        m_write_method = write_method;
        m_file_name = new_file_name;
        switch(m_write_method) {
            case 'a' :  m_write_method = 'a';
                        break;
            case 'w' :  m_write_method = 'w';
                        this -> clear(m_file_name);
        }   
        m_last_read = 'f';
        return *this;
    }

    // os.open(file_name).read()
    // os.popen(command).read()
    std::string read(char content = 'n'){
        content = (content == 'n') ? m_last_read : content;

        switch(content) {
            case 'f' :  return this -> read_file(); break;
            case 'c' :  return m_command; break;
            default  :  return "none"; 
        }    
    }

    std::string read_file(){
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

    OS write(std::string content = "", char write_method = 'n'){

        m_write_method = (write_method == 'n') ? m_write_method : write_method;

        switch(m_write_method) {
            case 'a' :  m_write_method = 'a';
                        m_file_data = this -> read() + content;
                        break;
            
            case 'w' :  m_write_method = 'w';
                        m_file_data = content;
                        this -> clear(m_file_name);
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

    OS clear(std::string content = ""){
        std::string file_to_wipe = (content == "") ? m_file_name : content;
        std::ofstream ofs;
        ofs.open(file_to_wipe, std::ofstream::out | std::ofstream::trunc);
        ofs.close();
        return *this;
    }

    OS remove(std::string content = ""){
        std::string file_to_delete = (content == "") ? m_file_name : content;
        ::remove( &file_to_delete[0] );
        return *this;
    }

    // ------------------------------------------

    vector<string> dir(string scan_start, vector<string> &vec_track, char mod1 = 'n', char mod2 = 'n'){
        
        // new_var = (condition) ? (value_if_true) : (value_if_false);
        char scope = (mod1 == 'r' || mod2 == 'r') ? scope = 'r' : scope = 'n';
        char include_folders = (mod1 == 'f' || mod2 == 'f') ? include_folders = 'f' : include_folders = 'n';

        DIR *current_dir = opendir (scan_start.c_str()); // starting dir given as a string
        // "dir_item" can be anything in the "current_dir" such as a new folder, file, binary, etc.
        while (struct dirent *dir_item_ptr = readdir(current_dir)){
            string dir_item =  (dir_item_ptr->d_name); // structure points to the getter to retrive the dir_item's name.
            if (dir_item != "." and dir_item != "./" and dir_item != ".."){
                if (dir_item_ptr->d_type == DT_DIR){
                    if(scope == 'r' ){
                        if(include_folders == 'f'){
                            vec_track.push_back(scan_start + "/" + dir_item);
                        }
                        this->dir(scan_start + "/"+ dir_item, vec_track, scope, include_folders); // recursive function
                    }
                }else if(dir_item == "read"){
                    break; // full dir already read, leave the loop
                }else{
                    vec_track.push_back(scan_start + "/"+ dir_item);
                }
            }
        }
        return vec_track;
        closedir (current_dir); 
    }

    bool file_exist(std::string file){
        return bool(std::ifstream(file));
    }

    // Replace popen and pclose with _popen and _pclose for Windows.
    OS popen(const std::string command){
        const char* commmand_ptr = &command[0];
        m_command.clear();
        int buf_size = 512;
        char buffer[buf_size];
        FILE* file = ::popen(commmand_ptr, "r");
        if (file){
            while (!feof(file)) {
                if (fgets(buffer, buf_size, file) != NULL)
                    m_command += buffer;
            }
        }else{
            cout << "Console Command Failed\n";
            cout << "-----------------------\n";
            cout << m_command << endl;
        }
        int returnCode = pclose(file);
        if(returnCode){ cout << "Shell Command Error Code: " << returnCode << endl; }

        m_last_read = 'c';
        return *this;
    }

    std::string operator[](int value){
        if (m_all_args.size() <= value){
            return blank;
        }else{
            return m_all_args[value];
        }
    }

    OS set_args(int argc, char** argv){
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
                        }else if(!key_exist(m_all_args[arg])){ // if the key doesn't already exist
                            m_args.insert(std::make_pair(m_all_args[arg], blank_vec));
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
    std::string file_data(){return m_file_data;}
    std::string file_name(){return m_file_name;}
    // -------------------------------------------------------------------------------
    std::string command()  {return m_command;}
    std::string read_command()  {return m_command;}
    std::string cmd()  {return m_command;}
    // -------------------------------------------------------------------------------
    std::vector<std::string> arg_vec(){return m_all_args;}
    std::string args_str(){return m_all_args_str;}
    bool if_arg(std::string find_arg){
        for(auto&arg : m_all_args){
            if (arg == find_arg)
                return true;
        return false;
        }
    }
    int arg_count(){return m_all_args.size();}

    std::vector<std::string> base_args(){return m_bargs;}
    std::string base_args_str(){return m_bargs_str;}

    std::vector< std::vector<std::string> > sub_args(){return m_sub_args;}
    std::string sub_args_str(){return m_sub_args_str;}
    // -------------------------------------------------------------------------------
    std::unordered_map<std::string, std::vector<std::string> > args(){return m_args;}
    
    std::vector<std::string> key(std::string key){return m_args.at(key);}
    
    std::string value(std::string key, int i){return m_args.at(key)[i];}

    std::vector<std::string> values(std::string key){
        std::vector<std::string> values = m_args.at(key);
        return values;
    }


    bool key_exist(std::string key){bool((m_args.find(key) != m_args.end()));};
    bool value_exist(std::string key, std::string value){
        for(size_t i = 0; i < m_args.at(key).size(); i++){
            if (m_args.at(key)[i] == value){
                return true;
            }
        }return false;
    }

    template<class T = std::string>
    void p(T str){cout << endl << "------\n" << str << endl;}
    void d(int i = 0){cout << endl << "---{dbg: " << i << "}---" << endl;}
    // -------------------------------------------------------------------------------
};


// TODO, make into multi-file and set instances properly
extern OS os; // Header
OS os;        // cpp

#endif
