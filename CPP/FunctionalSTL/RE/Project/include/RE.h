#pragma once

// lib: re
// version 1.4.0

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
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXIther express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/


#include<vector>
#include<string>
#include<regex>
#include<algorithm>
#include<iostream>

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #ifdef DLL_EXPORT
       #define EXI  __declspec(dllexport)
    #else
       #define EXI  __declspec(dllimport)
    #endif
#else
    #define EXI
#endif


namespace RE // Regular Expression
{
    // re:: does NOT take any vector inputs as an argugment
    // for that use ac:: for your array controller

    // =================================================================================================================================

    EXI bool Match(const std::string& in_pattern, const std::string& content);
    EXI bool MatchLine(const std::string& in_pattern, const std::string& content);
    EXI bool MatchAllLines(const std::string& in_pattern, const std::string& content);

    EXI bool Scan(const std::string& in_pattern, const std::string& content);
    EXI bool ScanLine(const std::string& in_pattern, const std::string& content);
    EXI bool ScanAllLines(const std::string& in_pattern, const std::string& content);

    // =================================================================================================================================

    EXI std::vector<std::string> Cont_Split(const std::string& in_pattern, const std::string& content);

    EXI std::vector<std::string> Split(const std::string& in_pattern, const std::string& content);
    EXI std::vector<std::string> Split(const std::string& in_pattern, const std::string&& content);

    EXI std::vector<std::string> Split(const char splitter, const std::string& content);
    EXI std::vector<std::string> Split(const char splitter, const std::string&& content);

    // =================================================================================================================================

    EXI bool HasNonAscii(const std::string& str);
    EXI std::string RemoveNonAscii(const std::string& str);

    // =================================================================================================================================

    EXI std::vector<std::string> Findwalk(const std::string& in_pattern, const std::string& content, const bool group = false);
    EXI std::vector<std::string> Findall(const std::string& in_pattern, const std::string& content, const bool group = false);
    EXI std::vector<std::string> Search(const std::string& in_pattern, const std::string& content, const bool group = false);

    // =================================================================================================================================

    EXI size_t Count(const char var_char, const std::string& input_str);
    EXI size_t Count(const std::string& in_pattern, const std::string& str);

    // =================================================================================================================================

    EXI std::string Sub(const std::string& in_pattern, const std::string& replacement, const std::string& content);

    EXI std::string Strip(const std::string& content);

    // =================================================================================================================================
}
