#ifndef _ORD_H_
#define _ORD_H_

#include<map>
#include<unordered_map>
#include<vector>
#include<string>
#include<algorithm>
#include<regex>


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


namespace ord
{

    // FUNCITONS <These all use Templates and must be fully in the '.h' file>
    // FUNCTIONS starting with "R" are to be used with R-values (not passed by reference)

    //    std::string               join(std::vector<T>& vec, std::string seperator = " ", bool tail = false){
    
    //    bool                      findKey(std::map<X, Y>& d_map, string key){
    //    std::vector<A>            keys(const std::map<A,B>& xmap){
    //    std::vector<B>            keyValues(const std::map<A,B>& xmap){
    //    void                      relational_copy(std::map<X,Y>& map1, std::map<X,Y>& map2, std::string key1, std::string new_keyname){


    //    bool                      findItem(const T& item, const std::vector<T>& vec){
    //    bool                      findMatch(const std::string& in_pattern, const std::vector<std::string>& vec){
    //    bool                      findSeg(const std::string& in_pattern, const std::vector<std::string>& vec){

    //    std::vector<std::string>  retMatches(const std::string& in_pattern, const std::vector<std::string>& vec){
    //    std::vector<std::string>  retSegs(const std::string& in_pattern, const std::vector<std::string>& vec){

    // =======================================================================================================

    template<typename A, typename B>
    std::vector<A> keys(const std::map<A,B>& xmap);
    template<typename A, typename B>
    std::vector<A> keys(const std::unordered_map<A,B>& xmap);
    template<typename T = std::string>
    bool findItem(const T& item, const std::vector<T>& vec);

    // =======================================================================================================
    // join vector items to a string
    template<typename T>
    std::string xjoin(std::vector<T>& vec, std::string seperator = " ", bool tail = false){
        std::ostringstream ostr;      
        for(T i : vec)
            ostr << i << seperator;
        if(tail == false)
            return (ostr.str().substr(0,ostr.str().size()-seperator.size()));
        return ostr.str();
    }
    template<typename T>
    std::string join(std::vector<T>& vec, std::string seperator = " ", bool tail = false){
        return xjoin<T>(vec, seperator, tail);
    }
    template<typename T>
    std::string Rjoin(std::vector<T> vec, std::string seperator = " ", bool tail = false){
        // r values can't be passed by reference
        return xjoin<T>(vec, seperator, tail);
    }
    // =======================================================================================================
    // map1 = {key1 , var1}
    // map2 = {var1,  var2}
    // output
    // map1 = {key1,   var1, "new_keyname", var2}

    // map1, map2, key1, new_keyname

    template<typename X, typename Y>
    void relational_copy(std::map<X,Y>& map1, std::map<X,Y>& map2, std::string key1, std::string new_keyname){
        if (findItem(map1.at(key1), keys(map2))){
            map1.insert({{new_keyname, map2.at(map1.at(key1))}});
        }
    }
    template<typename X, typename Y>
    void relational_copy(std::unordered_map<X,Y>& map1, std::unordered_map<X,Y>& map2, std::string key1, std::string new_keyname){
        if (findItem(map1.at(key1), keys(map2))){
            map1.insert({{new_keyname, map2.at(map1.at(key1))}});
        }
    }
    template<typename X, typename Y>
    void relational_copy(std::map<X,Y>& map1, std::unordered_map<X,Y>& map2, std::string key1, std::string new_keyname){
        if (findItem(map1.at(key1), keys(map2))){
            map1.insert({{new_keyname, map2.at(map1.at(key1))}});
        }
    }
    template<typename X, typename Y>
    void relational_copy(std::unordered_map<X,Y>& map1, std::map<X,Y>& map2, std::string key1, std::string new_keyname){
        if (findItem(map1.at(key1), keys(map2))){
            map1.insert({{new_keyname, map2.at(map1.at(key1))}});
        }
    }
    // =======================================================================================================
    // find Key > does it exist?
    template<typename X, typename Y>
    bool findKey(std::string key, std::unordered_map<X, Y>& d_map){
        return bool((d_map.find(key) != d_map.end()));
    }
    template<typename X, typename Y>
    bool RFindKey(std::string key, std::unordered_map<X, Y> d_map){
        return bool((d_map.find(key) != d_map.end()));
    }
    template<typename X, typename Y>
    bool findKey(std::string key, std::map<X, Y>& d_map){
        return bool((d_map.find(key) != d_map.end()));
    }
    template<typename X, typename Y>
    bool RFindKey(std::string key, std::map<X, Y> d_map){
        return bool((d_map.find(key) != d_map.end()));
    }
    // =======================================================================================================
    // ret vector of all keys from map
    template<typename A, typename B>
    std::vector<A> keys(const std::map<A,B>& xmap){
        std::vector<A> vec;
        for(typename std::map<A,B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
            vec.push_back(iter->first);
        return vec;
    }    
    template<typename A, typename B>
    std::vector<A> Rkeys(const std::map<A,B> xmap){
        std::vector<A> vec;
        for(typename std::map<A,B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
            vec.push_back(iter->first);
        return vec;
    }        // ret vector of all values from map
    template<typename A, typename B>
    std::vector<A> keys(const std::unordered_map<A,B>& xmap){
        std::vector<A> vec;
        for(typename std::unordered_map<A,B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
            vec.push_back(iter->first);
        return vec;
    }    
    template<typename A, typename B>
    std::vector<A> Rkeys(const std::unordered_map<A,B> xmap){
        std::vector<A> vec;
        for(typename std::unordered_map<A,B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
            vec.push_back(iter->first);
        return vec;
    }        // ret vector of all values from map
    template<typename A, typename B>
    std::vector<B> keyValues(const std::map<A,B>& xmap){
        std::vector<B> vec;
        for(typename std::unordered_map<A,B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
            vec.push_back(iter->second);
        return vec;
    }
    template<typename A, typename B>
    std::vector<B> RkeyValues(const std::map<A,B> xmap){
        std::vector<B> vec;
        for(typename std::unordered_map<A,B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
            vec.push_back(iter->second);
        return vec;
    }
    template<typename A, typename B>
    std::vector<B> keyValues(const std::unordered_map<A,B>& xmap){
        std::vector<B> vec;
        for(typename std::unordered_map<A,B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
            vec.push_back(iter->second);
        return vec;
    }
    template<typename A, typename B>
    std::vector<B> RkeyValues(const std::unordered_map<A,B> xmap){
        std::vector<B> vec;
        for(typename std::unordered_map<A,B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
            vec.push_back(iter->second);
        return vec;
    }
    // ============================================================================================================
    // Shorthand for std::find()
    template<typename T = std::string>
    bool findItem(const T& item, const std::vector<T>& vec){
        return (bool(std::find(vec.begin(), vec.end(), item)!=vec.end()));
    }
    template<typename T = std::string>
    bool RfindItem(const T& item, const std::vector<T> vec){
        return (bool(std::find(vec.begin(), vec.end(), item)!=vec.end()));
    }    
    bool findItem(char const* item, const std::vector<std::string>& vec){
        return (bool(std::find(vec.begin(), vec.end(), item)!=vec.end()));
    };   
    bool RfindItem(char const* item, const std::vector<std::string> vec){
        return (bool(std::find(vec.begin(), vec.end(), item)!=vec.end()));
    };
    // ============================================================================================================
    // regex match version of std::find()
    bool xfindMatch(const std::string& in_pattern, const std::vector<std::string>& vec);

    bool findMatch(const std::string& in_pattern, const std::vector<std::string>& vec){
        return xfindMatch(in_pattern, vec);
    }
    bool RfindMatch(const std::string in_pattern, const std::vector<std::string> vec){
        return xfindMatch(in_pattern, vec);
    }
    // ============================================================================================================
    // ret true if regex_search finds a match

    bool xfindSeg(const std::string& in_pattern, const std::vector<std::string>& vec);

    bool findSeg(const std::string& in_pattern, const std::vector<std::string>& vec){
        return xfindSeg(in_pattern, vec);
    }
    bool RfindSeg(const std::string& in_pattern, const std::vector<std::string> vec){
        return xfindSeg(in_pattern, vec);
    }
    // ============================================================================================================
    // return Matches found
    std::vector<std::string> xretMatches(const std::string& in_pattern, const std::vector<std::string>& vec);

    std::vector<std::string> retMatches(const std::string& in_pattern, const std::vector<std::string>& vec){
        return xretMatches(in_pattern, vec);
    }
    std::vector<std::string> RretMatches(const std::string in_pattern, const std::vector<std::string> vec){
        return xretMatches(in_pattern, vec);
    }
    // ============================================================================================================
    // return item from vector if a regex segment is found

    std::vector<std::string> xretSegs(const std::string& in_pattern, const std::vector<std::string>& vec);

    std::vector<std::string> retSegs(const std::string& in_pattern, const std::vector<std::string>& vec){
        return xretSegs(in_pattern, vec);
    }
    std::vector<std::string> RretSegs(const std::string& in_pattern, const std::vector<std::string> vec){
        return xretSegs(in_pattern, vec);
    }
    // ============================================================================================================


}

#endif
