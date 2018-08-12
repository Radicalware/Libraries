#ifndef _H_re_regex_
#define _H_re_regex_

#include<vector>
#include<string>
#include<regex>
#include<algorithm>

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



namespace re
{
    // re:: does NOT take any vector inputs as a namespace. 
    // To parse vectors/maps use ord:: which is short for order. 
    // Note, grouper/iter are not to be called directly.

    // ======================================================================================
    bool match(const std::string& in_pattern, const std::string& content);
    bool matchLine(const std::string& in_pattern, const std::string& content);
    bool matchLines(const std::string& in_pattern, const std::string& content);

    bool scan(const std::string& in_pattern, const std::string& content);
    bool scanLine(const std::string& in_pattern, const std::string& content);
    bool scanLines(const std::string& in_pattern, const std::string& content);
    // ======================================================================================

    std::vector<std::string> xsplit(const std::string& in_pattern, const std::string& content);

    std::vector<std::string> split(const std::string& in_pattern, const std::string& content);
    std::vector<std::string> split(const std::string& in_pattern, const std::string&& content);

    // ======================================================================================

    bool ASCII_check(const std::string& str);

    // re::search & re::findall use grouper/iterator, don't use them via the namespace directly
    std::vector<std::string> grouper(const std::string& content, std::vector<std::string>& ret_vector, const std::string& in_pattern);

    std::vector<std::string> iterator(const std::string& content, std::vector<std::string>& ret_vector, const std::string& in_pattern);

    // --------------------------------------------------------------------------------------
    std::vector<std::string> findall(const std::string& in_pattern, const std::string& content, const bool group = false);

    // ======================================================================================

    unsigned long char_count(const char var_char, const std::string& input_str);


    unsigned long count(const std::string& in_pattern, const std::string& str);

    // ======================================================================================

    std::string sub(const std::string& in_pattern, const std::string& replacement, const std::string& content);

    // ======================================================================================

}
#endif