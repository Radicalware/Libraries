
#include<map>
#include<unordered_map>
#include<vector>
#include<string>
#include<algorithm>
#include<regex>

#include "ord.h"


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

bool ord::xfindMatch(const std::string& in_pattern, const std::vector<std::string>& vec){
    std::regex pattern(in_pattern);
    std::vector<std::string>::const_iterator iter;
    for (iter = vec.begin(); iter != vec.end(); ++iter)
    {
        if (std::regex_match(*iter, pattern))
            return true;
    }
    return false;
}


bool ord::xfindImpression(const std::string& in_pattern, const std::vector<std::string>& vec){
    std::regex pattern(in_pattern);
    std::vector<std::string>::const_iterator iter;
    for (iter = vec.begin(); iter != vec.end(); ++iter)
    {
        if (std::regex_search(*iter, pattern)) // ret whole item so re.h for findall not needed
            return true;
    }
    return false;
}

std::vector<std::string> ord::xretMatches(const std::string& in_pattern, const std::vector<std::string>& vec){
    std::regex pattern(in_pattern);
    std::vector<std::string> ret_patterns;
    std::vector<std::string>::const_iterator iter;
    for (iter = vec.begin(); iter != vec.end(); ++iter)
    {
        if (std::regex_match(*iter, pattern))
            ret_patterns.push_back(*iter);
    }
    return ret_patterns;
}

std::vector<std::string> ord::xretImpressions(const std::string& in_pattern, const std::vector<std::string>& vec){
    std::regex pattern(in_pattern);
    std::vector<std::string> ret_patterns;
    std::vector<std::string>::const_iterator iter;
    for (iter = vec.begin(); iter != vec.end(); ++iter)
    {
        if (std::regex_search(*iter, pattern)) // ret whole item so re.h for findall not needed
            ret_patterns.push_back(*iter);
    }
    if (ret_patterns.size()){
    	return ret_patterns;
    }else{
    	ret_patterns.resize(1);
    	return ret_patterns;
    }
}