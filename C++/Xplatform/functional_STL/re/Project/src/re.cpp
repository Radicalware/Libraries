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
    std::regex rpattern(in_pattern);
    for (std::sregex_token_iterator iter(content.begin(), content.end(), rpattern, -1); iter != std::sregex_token_iterator(); ++iter)
        split_content.push_back(*iter);

    return split_content;
}

std::vector<std::string> re::split(const std::string& in_pattern, const std::string& content) {
    return re::cont_split(in_pattern, content);
}
std::vector<std::string> re::split(const std::string& in_pattern, const std::string&& content) {
    return re::cont_split(in_pattern, content);
}
std::vector<std::string> re::split(const char splitter, const std::string& content) {
    std::string str;
    str.insert(str.begin(), splitter);
    return re::cont_split(str, content);
};
std::vector<std::string> re::split(const char splitter, const std::string&& content) {
    std::string str;
    str.insert(str.begin(), splitter);
    return re::cont_split(str, content);
}

// ======================================================================================

bool re::match(const std::string& in_pattern, const std::string& content) {
    std::regex pattern(in_pattern);
    return bool(std::regex_match(content, pattern));
}
bool re::match_line(const std::string& in_pattern, const std::string& content) {
    std::vector<std::string> lines = re::split('\n', content);
    std::cout << "size: " << lines.size() << std::endl;
    std::regex pattern(in_pattern);
    for (std::vector<std::string>::const_iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_match(*iter, pattern)) {
            return true;
        }
        std::cout << '"' << *iter << '"' << std::endl;
    }
    return false;
}
bool re::match_lines(const std::string& in_pattern, const std::string& content) {
    std::vector<std::string> lines = re::split('\n', content);
    std::regex pattern(in_pattern);
    for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_match(*iter, pattern)) {
            std::cout << '"' << *iter << '"' << std::endl;
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

std::vector<std::string> re::findwalk(const std::string& in_pattern, const std::string& content, const bool group)
{
    std::vector<std::string> retv;
    std::regex rpattern(in_pattern);

    std::vector<std::string> lines = re::split('\n',content);
    for (const auto& line : lines) {
        for (std::sregex_token_iterator iter(line.begin(), line.end(), rpattern, 1); iter != std::sregex_token_iterator(); ++iter)
            retv.push_back(*iter);
    }

    return retv;
}

// --------------------------------------------------------------------------------------
std::vector<std::string> re::findall(const std::string& in_pattern, const std::string& content, const bool group /*=false*/)
{
    std::vector<std::string> retv;
    std::regex rpattern(in_pattern);

    for (std::sregex_token_iterator iter(content.begin(), content.end(), rpattern, 1); iter != std::sregex_token_iterator(); ++iter)
        retv.push_back(*iter);

    return retv;
}

std::vector<std::string> re::search(const std::string& in_pattern, const std::string& content, const bool group)
{
    std::vector<std::string> retv;
    std::regex rpattern(in_pattern);

    std::smatch matcher;
    if (std::regex_search(content, matcher, rpattern)) {
        size_t sz = matcher.size();
        for (std::smatch::const_iterator it = matcher.begin() + 1; it != matcher.end(); it++)
            retv.push_back(*it);
    }
    return retv;
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
