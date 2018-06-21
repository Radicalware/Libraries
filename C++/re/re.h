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

    // debugging code >>>
    // I left this here becaues I may make future updates
    void pt(const std::string& val){ std::cout << ">>> " << val << std::endl; }
    void pt(const char& val){ std::cout << ">>> " <<  val << std::endl; }
    void upt(const unsigned long& val){ std::cout << ">>> " <<  val << std::endl; }
    void fpt(const float& val){ std::cout << ">>> " <<  val << std::endl; }
    void pt(const int& val){ std::cout << ">>> " <<  val << std::endl; }    
    void pt(const double& val){ std::cout << ">>> " <<  val << std::endl; }
    void db(){std::cout << "\n********************\n";}
    void lf(){std::cout << '\n';}
    // debugging code <<

    std::vector<std::string> split(const std::string& in_pattern, const std::string& content){
        std::vector<std::string> split_content;
        std::regex pattern(in_pattern);
        copy( std::sregex_token_iterator(content.begin(), content.end(), pattern, -1),
        std::sregex_token_iterator(),back_inserter(split_content));  
        return split_content;
    }
    // ======================================================================================
    // re::search & re::findall use grouper/iterator, don't use them via the namespace directly
    std::vector<std::string>grouper(const std::string& content, std::vector<std::string>& ret_vector, const std::string& in_pattern)
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


    std::vector<std::string>iterator(const std::string& content, std::vector<std::string>& ret_vector, const std::string& in_pattern){
        std::smatch match_array;
        std::regex pattern(in_pattern);
        for(std::sregex_iterator iter_index = std::sregex_iterator(content.begin(), content.end(), pattern);
                                 iter_index != std::sregex_iterator(); ++iter_index)
        {
            match_array = *iter_index;
            for(auto index = 1; index < match_array.size(); ++index ){
                if (!match_array[index].str().empty()) {
                    // regex found for a line/element in the arrays
                    ret_vector.push_back(match_array[index]); 
                    break;
                }
            }
        }
        return ret_vector;
    }

    // --------------------------------------------------------------------------------------
    std::vector<std::string> findall(const std::string& in_pattern, const std::string& content, const char group = false)
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
                ret_vector = iterator(line, ret_vector, in_pattern);
            }
        }
        else // If you chose grouping, you have more controle but more work. (C++ not Python style)
        {
            for (int i = 0; i < new_line_count; i++ )
            {
                // made a copy of the target line
                ret_vector = grouper(split_string[i], ret_vector, in_pattern);
            }
        }

        return ret_vector;
    }
    // ======================================================================================

    std::vector<std::string> search(const std::string& in_pattern, const std::string& content, const char group = false)
    {
        std::smatch array;
        std::vector<std::string> ret_vector;
        if (group == false){
            ret_vector = iterator(content, ret_vector, in_pattern);
        }else{
            ret_vector = grouper(content, ret_vector, in_pattern);
        }
        return ret_vector;
    }
    // ======================================================================================

    double char_count(const char var_char, const std::string& input_str)
    {
        double n = std::count(input_str.begin(), input_str.end(), var_char);
        return n;
    }

    double str_count(const std::string& in_pattern, const std::string& str)
    {
        std::vector<std::string> matches = search(in_pattern, str);
        double n = matches.size();
        return n;
    }

    // ======================================================================================


    bool match(const std::string& in_pattern, const std::string& content)
    {
        std::regex pattern(in_pattern);
        return bool(std::regex_match(content, pattern));
    }
    // ======================================================================================

    std::string sub(const std::string& in_pattern, const std::string& replacement, const std::string& content)
    {
        std::regex pattern(in_pattern);
        return std::regex_replace(content, pattern, replacement);
    }
    // ======================================================================================

    std::string slice(const std::string& content, double x = 0, double y = 0, double z = 0)
    {
        // python based trimming method >> string[num:num:num]

        // Currently this only works with strings
        // I will add the functionality for it to handle arrays/vectors later
        double len = content.length();


        std::string sliced;

        if(x == 0 and y == 0 and z == 0){
            return content;
        }

        if (y == 0 && z >= 0){
            y = content.length();
        }

        if (x < 0){
            x = content.length() + x;
        }

        if (y < 0){
            y = content.length() + y;
        }

        if(z == 0){
            z = 1;
        }

        double sliced_size;
        double abs_z;
        if (z < 0){
            abs_z = z * -1;
        }
        if(x > y){
            sliced_size = (((x - y) / abs_z) + 2);
        }
        sliced.resize(sliced_size);
        double idx = x;
        if (z >= 0){
            do{
                if(idx >= y){
                    break;
                }
                sliced[idx] = content[idx];
                idx += z;
            }while(true);
        }else{
            do{
                if(idx <= y){
                    break;
                }
                sliced[idx] = content[idx];
                idx += z;
            }while(true);
        }

        return sliced;
        
    }
}
