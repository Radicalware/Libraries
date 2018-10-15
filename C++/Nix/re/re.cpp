#include<vector>
#include<string>
#include<regex>
#include<algorithm>

#include "./re.h"

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


// ======================================================================================

std::vector<std::string> re::xsplit(const std::string& in_pattern, const std::string& content){
    std::vector<std::string> split_content;
    std::regex pattern(in_pattern);
    copy( std::sregex_token_iterator(content.begin(), content.end(), pattern, -1),
    std::sregex_token_iterator(),back_inserter(split_content));  
    return split_content;
}

std::vector<std::string> re::split(const std::string& in_pattern, const std::string& content){
    return re::xsplit(in_pattern, content);
}
std::vector<std::string> re::split(const std::string& in_pattern, const std::string&& content){
    return re::xsplit(in_pattern, content);
}

// ======================================================================================

bool re::match(const std::string& in_pattern, const std::string& content){
    std::regex pattern(in_pattern);
    return bool(std::regex_match(content, pattern));
}
bool re::matchLine(const std::string& in_pattern, const std::string& content){
	std::vector<std::string> lines = re::split("\n", content);
    std::regex pattern(in_pattern);
	for(std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++){
		if(std::regex_match(*iter, pattern)){
			return true;
		}
	}
	return false;
}
bool re::matchLines(const std::string& in_pattern, const std::string& content){
	std::vector<std::string> lines = re::split("\n", content);
    std::regex pattern(in_pattern);
	for(std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++){
		if(!std::regex_match(*iter, pattern)){
			return false;
		}
	}
	return true;
}

bool re::scan(const std::string& in_pattern, const std::string& content){
    std::regex pattern(in_pattern);
    return bool(std::regex_search(content, pattern));
}
bool re::scanLine(const std::string& in_pattern, const std::string& content){
	std::vector<std::string> lines = re::split("\n", content);
    std::regex pattern(in_pattern);
	for(std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++){
		if(std::regex_search(*iter, pattern)){
			return true;
		}
	}
	return false;
}
bool re::scanLines(const std::string& in_pattern, const std::string& content){
	std::vector<std::string> lines = re::split("\n", content);
    std::regex pattern(in_pattern);
	for(std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++){
		if(!std::regex_search(*iter, pattern)){
			return false;
		}
	}
	return true;
}


// ======================================================================================
bool re::ASCII_check(const std::string& str) {
  for (auto& c: str) {
    if (static_cast<unsigned char>(c) > 127) {
      return false;
    }
  }
  return true;
}

// re::search & re::findall use grouper/iterator, don't use them via the namespace directly
std::vector<std::string> re::grouper(const std::string& content, std::vector<std::string>& ret_vector, const std::string& in_pattern)
{
    // note: passing ret_vector by reference caused memory corruption (hence I passed by value)
    std::smatch match_array;
    std::regex pattern(in_pattern);
    std::string::const_iterator searchStart( content.cbegin() );
    std::string::const_iterator prev( content.cbegin() );
    while ( regex_search( searchStart, content.cend(), match_array, pattern ) )
    {
        for(int i = 0; i < match_array.size(); i++){
            ret_vector.push_back(match_array[i]);
        }

        searchStart += match_array.position() + match_array.length();
        if (searchStart == prev){ break;
        }else{ prev = searchStart; }
    }
    return ret_vector;
}


std::vector<std::string> re::iterator(const std::string& content, std::vector<std::string>& ret_vector, const std::string& in_pattern){
    std::smatch match_array;
    std::regex pattern(in_pattern);
    int start_iter = 0;
    if (match("^.*\\?\\:.*$",in_pattern) == true){
        start_iter = 1;
    }
    for(std::sregex_iterator iter_index = std::sregex_iterator(content.begin(), content.end(), pattern);
                             iter_index != std::sregex_iterator(); ++iter_index)
    {
        match_array = *iter_index;
        for(auto index = start_iter; index < match_array.size(); ++index ){
            if (!match_array[index].str().empty()) {
                // regex found for a line/element in the arrays
                // cout << "** " << match_array[index] << '\n';
                ret_vector.push_back(match_array[index]); 
                
            }
        }
    }
    return ret_vector;
}

// --------------------------------------------------------------------------------------
std::vector<std::string> re::findall(const std::string& in_pattern, const std::string& content, const bool group /*=false*/)
{
    std::vector<std::string> ret_vector;
    std::vector<std::string> split_string;

    int new_line_count = std::count(content.begin(), content.end(), '\n');

    int split_loc;
    std::string tmp_content = content;
    // if/else: set each line to an element of the split_string vector
    if ((new_line_count > 0 && new_line_count < content.length()) && new_line_count != 0)
    {  
        split_string.resize(new_line_count+1);

        for(int i = 0; i < new_line_count; i++){
            split_loc = tmp_content.find('\n');
            split_string[i] = tmp_content.substr(0,split_loc);
            tmp_content = tmp_content.substr(split_loc+1, tmp_content.length() - split_loc-1);
        }
    }
    else
    {
        new_line_count = 1;
        split_string.push_back(content);
    }

    std::string line;
    std::smatch match_array;
    std::regex pattern(in_pattern);
    // now iterate through each line (now each element of the array)
    if (group == false){ // grouping is set to false by default
        for (int index = 0; index < new_line_count; index++ )
        {
            line = split_string[index].substr(0,split_string[index].length());
            // Make a copy of
            iterator(line, ret_vector, in_pattern);
        }
    }
    else // If you chose grouping, you have more controle but more work. (C++ not Python style)
    {
        for (int i = 0; i < new_line_count; i++ )
        {
            // made a copy of the target line
            grouper(split_string[i], ret_vector, in_pattern);
        }
    }

    std::vector<std::string> filtered;
    for(int i = 0; i < ret_vector.size(); i++){
        // ASCII_check(ret_vector[i]) && 
        if (ret_vector[i].size() > 1 && match("^.*[^\\s].*$",ret_vector[i])){
            filtered.push_back(ret_vector[i]);
        }
    }

    return filtered;
}

// ======================================================================================

unsigned long re::char_count(const char var_char, const std::string& input_str)
{
    unsigned long n = std::count(input_str.begin(), input_str.end(), var_char);
    return n;
}

unsigned long re::count(const std::string& in_pattern, const std::string& str)
{
    std::vector<std::string> matches = findall(in_pattern, str);
    unsigned long n = matches.size();
    return n;
}


// ======================================================================================

std::string re::sub(const std::string& in_pattern, const std::string& replacement, const std::string& content)
{
    std::regex pattern(in_pattern);
    return std::regex_replace(content, pattern, replacement);
}


