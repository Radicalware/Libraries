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
	std::string m_file_name;
	std::string m_file_data;

	std::string m_command;

	int    m_argc;
	char** m_argv;
	
	std::unordered_map<std::string, std::vector<std::string> > m_args; // {base_arg, sub_args}

	std::vector<std::string> m_all_args; 
	std::string m_args_str;

	std::vector<std::string> m_bargs; // base args
	std::string m_bargs_str;
	std::vector< std::vector<std::string> > m_sub_args; // sub args
	std::string m_sub_args_str;

public:

	OS(int c_argc, char** c_argv):
	m_argc(c_argc), m_argv(c_argv)
	{(this)->set_args(m_argc, m_argv);};


	OS(){};
	~OS(){};
	// ------------------------------------------
	OS open(std::string new_file){
		m_file_name = new_file;
		return *this;
	}

	OS write(std::string content){
		const char* content_ptr = &content[0];
		std::ofstream os_file(m_file_name);
		if (os_file.is_open()){
			os_file << content_ptr;
			os_file.close();
		}
		return *this;
	}

	std::string read(){
		std::ifstream os_file(m_file_name);
		std::string line;
		std::string m_file_data = "";

		std::string file_content;
		if(os_file.is_open()){
			while (getline(os_file,line)){
				m_file_data += line + '\n';
			}
			os_file.close();
		}
		return m_file_data;
	}
	// ------------------------------------------

	vector<string> dir(string scan_start, string scope, string include_folder, vector<string> &vec_track){
		
		DIR *current_dir = opendir (scan_start.c_str()); // starting dir given as a string
		// "dir_item" can be anything in the "current_dir" such as a new folder, file, binary, etc.
		while (struct dirent *dir_item_ptr = readdir(current_dir)){
			string dir_item =  (dir_item_ptr->d_name); // structure points to the getter to retrive the dir_item's name.
			if (dir_item != "." and dir_item != "./" and dir_item != ".."){
				if (dir_item_ptr->d_type == DT_DIR){
					if(scope == "r" or scope == "-r"){
						if(include_folder == "f" or include_folder == "-f"){
							vec_track.push_back(scan_start + "/" + dir_item);
						}
						this->dir(scan_start + "/"+ dir_item, scope, include_folder, vec_track); // recursive function
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

	// Replace popen and pclose with _popen and _pclose for Windows.
	std::string console(const std::string command){
		const char* commmand_ptr = &command[0];
		m_command = command;
		int buf_size = 512;
		char buffer[buf_size];
		std::string result = "";
		FILE* file = popen(commmand_ptr, "r");
		if (file){
		    while (!feof(file)) {
		        if (fgets(buffer, buf_size, file) != NULL)
		            result += buffer;
		    }
		}else{
			cout << "Console Command Failed\n";
			cout << "-----------------------\n";
			cout << m_command << endl;
		}
		int returnCode = pclose(file);
		if(returnCode){ cout << "Shell Command Error Code: " << returnCode << endl; }

		return result;
	}

	OS set_args(int argc, char** argv){
		m_all_args.resize(argc);
		bool sub_arg = false;
		bool first_header = false;
		int first_sub = 0;
		std::vector<std::string> prep_sub_arg;
		for(int i = 0; i < argc; i++){
			
			m_all_args[i] = argv[i];
			m_args_str += m_all_args[i] + ' ';
			if (m_all_args[i][0] == '-' or m_all_args[i][0] == '/'){
				m_bargs.push_back(m_all_args[i]);
				m_bargs_str += m_all_args[i] + ' ';
				first_header = true;
				sub_arg = false;
				first_sub += 1;
			}else{
				sub_arg = true;
				if(first_header){first_sub += 1;}
			}

			if(first_header && sub_arg and (i != argc-1)){
				prep_sub_arg.push_back(m_all_args[i]);
				m_sub_args_str += m_all_args[i] + " ";
			}else if(first_header && first_sub >= 2 && (sub_arg == false || (i == argc-1))) {
				if(i == argc-1){
					prep_sub_arg.push_back(m_all_args[i]);
					m_sub_args_str += m_all_args[i];
				}
				m_sub_args.push_back(prep_sub_arg);
				prep_sub_arg.clear();
				m_sub_args_str += '\n';
			}
		}

		if(first_header){
			for(int i = 0; i < m_bargs.size(); i++){
				m_args.insert(std::make_pair(m_bargs[i],m_sub_args[i]));
			}
		} 
		return *this;	
	}


	// -------------------------------------------------------------------------------
	std::string file_data(){return m_file_data;}
	std::string file_name(){return m_file_name;}
	// -------------------------------------------------------------------------------
	std::string command()  {return m_command;}
	// -------------------------------------------------------------------------------
	std::vector<std::string> arg_vec(){return m_all_args;}
	std::string args_str(){return m_args_str;}

	std::vector<std::string> base_args(){return m_bargs;}
	std::string base_args_str(){return m_bargs_str;}

	std::vector< std::vector<std::string> > sub_args(){return m_sub_args;}
	std::string sub_args_str(){return m_sub_args_str;}
	// -------------------------------------------------------------------------------
	std::unordered_map<std::string, std::vector<std::string> > dict(){return m_args;}
	
	std::vector<std::string> key(std::string key){return m_args.at(key);}
	std::string value(std::string key, int i){return m_args.at(key)[i];}
	
	bool if_key(std::string key){
		for(size_t i = 0; i < m_bargs.size(); i++){
			if(key == m_bargs[i]){
				return true;
			}
		}return false;
	};
	bool if_value(std::string key, std::string value){
		for(size_t i = 0; i < m_args.at(key).size(); i++){
			if (m_args.at(key)[i] == value){
				return true;
			}
		}return false;
	}
	// -------------------------------------------------------------------------------
};
