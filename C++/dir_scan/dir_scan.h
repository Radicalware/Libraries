#include <dirent.h> 
#include <vector>
#include <string>

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

using std::string;
using std::vector;

// This is a good method because we can get all the files (with or without folder) recursive or not
// from a target directory, and then have all those elements placed into a vector for easy 'for'
// loop usage and functional parsing.

vector<string> scan_dir(string scan_start, string scope, string include_folder, vector<string> &vec_track){
	
	DIR *current_dir = opendir (scan_start.c_str()); // starting dir given as a string

	// "dir_item" can be anything in the "current_dir" such as a new folder, file, binary, etc.

	while (struct dirent *dir_item_ptr = readdir(current_dir)){
		string dir_item =  (dir_item_ptr->d_name); // structure points to the getter to retrive the dir_item's name.
		if (dir_item != "." and dir_item != "./" and dir_item != ".."){
			if (dir_item_ptr->d_type == DT_DIR){
				if(scope == "r" or scope == "-r"){
					if(include_folder == "f" or include_folder == "-f"){
						vec_track.push_back(dir_item);
					}
					scan_dir(scan_start + "/"+ dir_item, scope, include_folder, vec_track); // recursive function
				}
			}else if(dir_item == "read"){
				break; // full dir already read, leave the loop
			}else{
				vec_track.push_back(dir_item);
			}
		}
	}
	return vec_track;
	closedir (current_dir);	
}


