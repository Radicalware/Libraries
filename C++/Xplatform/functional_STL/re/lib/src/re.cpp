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


#include "re.h"

// ======================================================================================


std::vector<std::string> re::cont_split(const std::string& in_pattern, const std::string& content) {
    std::vector<std::string> split_content;
    std::regex pattern(in_pattern);
    std::copy(std::sregex_token_iterator(content.begin(), content.end(), pattern, -1),
        std::sregex_token_iterator(), back_inserter(split_content));
    return split_content;
}

std::vector<std::string> re::split(const std::string& in_pattern, const std::string& content) {
    return re::cont_split(in_pattern, content);
}
std::vector<std::string> re::split(const std::string& in_pattern, const std::string&& content) {
    return re::cont_split(in_pattern, content);
}

std::vector<std::string> re::cont_split(const char splitter, const std::string& content) {
    std::vector<std::string> all_sections;
    std::string current_section;
    for (std::string::const_iterator it = content.begin(); it < content.end(); it++) {
        if (*it == splitter) {
            if (current_section.size()) {
                all_sections.push_back(current_section);
                current_section.clear();
            }
        } else {
            current_section += *it;
        }
    }
    if (current_section.size())
        all_sections.push_back(current_section);
    return all_sections;
}

std::vector<std::string> re::split(const char splitter, const std::string& content) {
    return re::cont_split(splitter, content);
};
std::vector<std::string> re::split(const char splitter, const std::string&& content) {
    return re::cont_split(splitter, content);
};

// ======================================================================================

bool re::match(const std::string& in_pattern, const std::string& content) {
    std::regex pattern(in_pattern);
    return bool(std::regex_match(content, pattern));
}
bool re::match_line(const std::string& in_pattern, const std::string& content) {
    std::vector<std::string> lines = re::split('\n', content);
    std::regex pattern(in_pattern);
    for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_match(*iter, pattern)) {
            return true;
        }
    }
    return false;
}
bool re::match_lines(const std::string& in_pattern, const std::string& content) {
    std::vector<std::string> lines = re::split('\n', content);
    std::regex pattern(in_pattern);
    for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_match(*iter, pattern)) {
            return false;
        }
    }
    return true;
}

bool re::scan(const std::string& in_pattern, const std::string& content) {
    std::regex pattern(in_pattern);
    return bool(std::regex_search(content, pattern));
}
bool re::scan_line(const std::string& in_pattern, const std::string& content) {
    std::vector<std::string> lines = re::split('\n', content);
    std::regex pattern(in_pattern);
    for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_search(*iter, pattern)) {
            return true;
        }
    }
    return false;
}
bool re::scan_lines(const std::string& in_pattern, const std::string& content) {
    std::vector<std::string> lines = re::split('\n', content);
    std::regex pattern(in_pattern);
    for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_search(*iter, pattern)) {
            return false;
        }
    }
    return true;
}


// ======================================================================================
bool re::has_non_ascii(const std::string& str) {
    for (char c : str) {
        if (!((static_cast<unsigned char>(c) > 0) || (static_cast<unsigned char>(c) < 128))) {
            return true;
        }
    }
    return false;
}

std::string re::remove_non_ascii(const std::string& data) {
    std::string clean_data;
    clean_data.reserve(data.size());
    for (std::string::const_iterator it = data.begin(); it < data.end(); it++) {
        if (int(*it) > 0 && int(*it) < 128)
            clean_data += *it;
    }
    return clean_data;
}

// re::search & re::findall use grouper/iterator, don't use them via the namespace directly
std::vector<std::string> re::grouper(const std::string& content, std::vector<std::string>& ret_vector, const std::string& in_pattern) {
    // note: passing ret_vector by reference caused memory corruption (hence I passed by value)
    std::smatch match_array;
    std::regex pattern(in_pattern);
    std::string::const_iterator searchStart(content.cbegin());
    std::string::const_iterator prev(content.cbegin());
    while (regex_search(searchStart, content.cend(), match_array, pattern)) {
        for (int i = 0; i < match_array.size(); i++) {
            ret_vector.push_back(match_array[i]);
        }

        searchStart += match_array.position() + match_array.length();
        if (searchStart == prev) {
            break;
        } else { prev = searchStart; }
    }
    return ret_vector;
}


std::vector<std::string> re::iterator(const std::string& content, std::vector<std::string>& ret_vector, const std::string& in_pattern) {
    //std::smatch match_array;
    std::regex pattern(in_pattern);
    int start_iter = 1;
    if (match(R"(^(.+?)(\(\?\:)(.+?)$)", in_pattern) == true) {
        start_iter = 2;
    }
    std::sregex_iterator iter_index = std::sregex_iterator(content.begin(), content.end(), pattern);
    for (iter_index; iter_index != std::sregex_iterator(); ++iter_index) {
		std::match_results<std::string::const_iterator> match_array(*iter_index);
        //match_array = *iter_index;
		//std::sregex_iterator match_array = iter_index;
        for (int index = start_iter; index < match_array.size(); ++index) {
            if (!match_array[index].str().empty()) {
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

    size_t new_line_count = std::count(content.begin(), content.end(), '\n');
	size_t split_loc;
    std::string tmp_content = content;
    // if/else: set each line to an element of the split_string vector
    if (new_line_count) {

        for (int i = 0; i < new_line_count; i++) {
            split_loc = tmp_content.find('\n');
            split_string.push_back(tmp_content.substr(0, split_loc));
            tmp_content = tmp_content.substr(split_loc + 1, tmp_content.length() - split_loc - 1);
        }
    } else {
        new_line_count = 1;
        split_string.push_back(content);
    }

    std::smatch match_array;
    std::regex pattern(in_pattern);
    // now iterate through each line (now each element of the array)
    if (group == false) { // grouping is set to false by default
        for (int index = 0; index < new_line_count; index++) {
            iterator(split_string[index], ret_vector, in_pattern);
        }
    } else { // If you chose grouping, you have more controle but more work. (C++ not Python style)
        for (int i = 0; i < new_line_count; i++) {
            grouper(split_string[i], ret_vector, in_pattern);
        }
    }

    std::vector<std::string> filtered;
    for (int i = 0; i < ret_vector.size(); i++) {
        if (!(ret_vector[i].size() == 1 && ret_vector[i].c_str()[0] == ' '))
            filtered.push_back(ret_vector[i]);
    }

    return filtered;
}

// ======================================================================================

size_t re::count(const char var_char, const std::string& input_str) {
	return std::count(input_str.begin(), input_str.end(), var_char);
}

size_t re::count(const std::string& in_pattern, const std::string& str) {
	return re::split(in_pattern, str).size();
}

// ======================================================================================

std::string re::sub(const std::string& in_pattern, const std::string& replacement, const std::string& content) {
    std::regex pattern(in_pattern);
    return std::regex_replace(content, pattern, replacement);
}

std::string re::strip(const std::string& content) {
    return re::sub(R"(^\s+|\s+$)", "", content);
}
