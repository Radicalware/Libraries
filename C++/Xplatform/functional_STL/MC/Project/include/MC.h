#pragma once

// MC:: version v1.1.1

/*
* Copyright[2018][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYcont_oOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file econt_cept in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either econt_press or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/


#include<map>
#include<unordered_map>
#include<vector>
#include<string>
#include<algorithm>
#include<regex>
#include<sstream>


namespace MC // map controller
{

    // std::vector<K> all_keys      (const std::map<K, V>& i_map)
    // std::vector<V> all_values    (const std::map<K, V>& i_map)
    // V              key_data      (const K& key_i, const std::map<K, V>& i_map)

    // bool           has_key       (const K& key, const M& d_map)
    // bool           has_key_value (const K& key, const EV& value, const M& d_map)

    // void           cont_copy         (M1& map1, const k new_keyname, const M2& map2, const v2 key1)

    template<typename K, typename V>
    std::vector<K> GetAllKeys(const std::map<K, V>& i_map);

    template<typename K, typename V>
    std::vector<K> GetAllKeys(const std::unordered_map<K, V>& i_map);

    // =================================================================================================================================
    // find Key > does it econt_ist?

    template<typename K, typename M>
    inline bool Cont_HasKey(const K& key, const M& d_map) {
        if (bool(d_map.find(key) != d_map.end())) {
            if (d_map.at(key).size() > 0) {
                return true;
            }
            return false;
        }
        return false;
    }
    template<typename K, typename M>
    inline bool HasKey(char* const i_key, const M& d_map) {
        std::string Key(i_key);
        return Cont_HasKey(Key, d_map);
    }
    template<typename K, typename M>
    inline bool HasKey(const K& key, const M& d_map) {
        return Cont_HasKey(key, d_map);
    }
    template<typename K, typename M>
    inline bool HasKey(const K& key, const M&& d_map) {
        return Cont_HasKey(key, d_map);
    }
    // =================================================================================================================================
    // if you have a vector for your value
    // this will recursivly search to see if you have that value
    // if you have a single value for your value just do "if(value == map.at(value))"

    template<typename K, typename EV, typename M>
    inline bool Cont_HasKeyValue(const K& key, const EV& value, const M& d_map) {
        if (bool(d_map.find(key) != d_map.end())) {
            if (d_map.at(key).size() > 0) {
                for (auto iter = d_map.at(key).begin(); iter != d_map.at(key).end(); iter++) {
                    if (*iter == value)
                        return true;
                }
                return false;
            }
            return false;
        }
        return false;
    }
    // Key, Iterator-Type, Element-Value, Map
    template<typename K, typename EV, typename M>
    inline bool HasKeyValue(char* const i_key, char* const i_value, const M& d_map) {
        std::string key = i_key;
        std::string value = i_value;
        return Cont_HasKeyValue(key, value, d_map);
    }
    template<typename K, typename EV, typename M>
    inline bool HasKeyValue(const K& key, const EV& value, const M& d_map) {
        return Cont_HasKeyValue(key, value, d_map);
    }

    template<typename K, typename EV, typename M>
    inline bool HasKeyValue(char* const i_key, const EV& value, const M& d_map) {
        std::string key = i_key;
        return Cont_HasKeyValue(key, value, d_map);
    }
    template<typename K, typename EV, typename M>
    inline bool HasKeyValue(const K& key, char* const i_value, const M& d_map) {
        std::string value = i_value;
        return Cont_HasKeyValue(key, value, d_map);
    }

    // =================================================================================================================================
    // ret vector of all keys from map

    template<typename K, typename V>
    inline std::vector<K> GetAllKeys(const std::map<K, V>& i_map) {
        std::vector<K> vec;
        for (typename std::map<K, V>::const_iterator iter = i_map.begin(); iter != i_map.end(); ++iter)
            vec.push_back(iter->first);
        return vec;
    }
    template<typename K, typename V>
    inline std::vector<K> GetAllKeys(const std::map<K, V>&& i_map) {
        std::vector<K> vec;
        for (typename std::map<K, V>::const_iterator iter = i_map.begin(); iter != i_map.end(); ++iter)
            vec.push_back(iter->first);
        return vec;
    }        // ret vector of all values from map
    template<typename K, typename V>
    inline std::vector<K> GetAllKeys(const std::unordered_map<K, V>& i_map) {
        std::vector<K> vec;
        for (typename std::unordered_map<K, V>::const_iterator iter = i_map.begin(); iter != i_map.end(); ++iter)
            vec.push_back(iter->first);
        return vec;
    }
    template<typename K, typename V>
    inline std::vector<K> GetAllKeys(const std::unordered_map<K, V>&& i_map) {
        std::vector<K> vec;
        for (typename std::unordered_map<K, V>::const_iterator iter = i_map.begin(); iter != i_map.end(); ++iter)
            vec.push_back(iter->first);
        return vec;
    }
    // =================================================================================================================================
    // return all values for every key in a vector 

    template<typename K, typename V>
    inline std::vector<V> GetAllValues(const std::map<K, V>& i_map) {
        std::vector<V> vec;
        for (typename std::unordered_map<K, V>::const_iterator iter = i_map.begin(); iter != i_map.end(); ++iter)
            vec.push_back(iter->second);
        return vec;
    }
    template<typename K, typename V>
    inline std::vector<V> GetAllValues(const std::map<K, V>&& i_map) {
        std::vector<V> vec;
        for (typename std::unordered_map<K, V>::const_iterator iter = i_map.begin(); iter != i_map.end(); ++iter)
            vec.push_back(iter->second);
        return vec;
    }
    template<typename K, typename V>
    inline std::vector<V> GetAllValues(const std::unordered_map<K, V>& i_map) {
        std::vector<V> vec;
        for (typename std::unordered_map<K, V>::const_iterator iter = i_map.begin(); iter != i_map.end(); ++iter)
            vec.push_back(iter->second);
        return vec;
    }
    template<typename K, typename V>
    inline std::vector<V> GetAllValues(const std::unordered_map<K, V>&& i_map) {
        std::vector<V> vec;
        for (typename std::unordered_map<K, V>::const_iterator iter = i_map.begin(); iter != i_map.end(); ++iter)
            vec.push_back(iter->second);
        return vec;
    }

    // =================================================================================================================================
    // return the key's data
    // 1st does the key econt_ist
    // if (yes) return the data; else return V()

    template<typename S, typename K, typename V>
    inline V GetKeyData(char* const key_i, const std::map<K, V>& i_map) {
        std::string key = key_i;
        if (MC::Cont_HasKey(key_i, i_map)) 
            return i_map.at(key_i);
        return V();
    }
    template<typename S, typename K, typename V>
    inline V GetKeyData(const S& key_i, const std::map<K, V>& i_map) {
        if (MC::Cont_HasKey(key_i, i_map))
            return i_map.at(key_i);
        return V();
    }
    template<typename S, typename K, typename V>
    inline V GetKeyData(const S& key_i, const std::unordered_map<K, V>& i_map) {
        if (MC::Cont_HasKey(key_i, i_map)) 
            return i_map.at(key_i);
        return V();
    }
    template<typename S, typename K, typename V>
    inline V GetKeyData(const S&& key_i, const std::map<K, V>& i_map) {
        if (MC::Cont_HasKey(key_i, i_map)) 
            return i_map.at(key_i);
        return V();
    }
    template<typename S, typename K, typename V>
    inline V GetKeyData(const S&& key_i, const std::unordered_map<K, V>& i_map) {
        if (MC::Cont_HasKey(key_i, i_map)) 
            return i_map.at(key_i);
        return V();
    }
    // =================================================================================================================================
    // cross copy aka xcopy aka relational copy

    template<typename M1, typename M2, typename k = std::string, typename v2 = std::string>
    inline void XCopy(M1& map1, const k new_keyname, const M2& map2, const v2 key1) {
        std::vector<std::string> map2_keys = MC::GetAllKeys(map2);
        if (std::find(map2_keys.begin(), map2_keys.end(), key1) == map2_keys.end()) {
            map1.insert({ {new_keyname, map2.at(map1.at(key1))} });
        }
    }
    // =================================================================================================================================
}
