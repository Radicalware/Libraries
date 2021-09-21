#pragma once

// ac:: version v1.0.1

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


#include<vector>
#include<string>
#include<algorithm>
#include<regex>
#include<sstream>

namespace AC // Array Type Controller for (vector, deque, set, etc.)
{

    // =================================================================================================================================
    // join vector items to a string
    template<typename T>
    std::string Cont_Join(const T& vec, const std::string seperator = "", bool tail = false) {
        std::ostringstream ostr;
        for (auto& i : vec)
            ostr << i << seperator;
        if (tail == false)
            return (ostr.str().substr(0, ostr.str().size() - seperator.size()));
        return ostr.str();
    }
    template<typename T>
    std::string Join(const T& vec, const std::string seperator = "", bool tail = false) {
        return Cont_Join<T>(vec, seperator, tail);
    }
    template<typename T>
    std::string Join(const T&& vec, const std::string seperator = "", bool tail = false) {
        // r values can't be passed by reference
        return Cont_Join<T>(vec, seperator, tail);
    }
    template<typename T>
    std::string Cont_PtrJoin(const T& vec, const std::string seperator = "", bool tail = false) {
        std::string str;
        for (auto it = vec.begin(); it != vec.end(); it++)
            str += **it + seperator;

        if (tail == false)
            return (str.substr(0, str.size() - seperator.size()));
        return str;
    }
    template<typename T>
    std::string PtrJoin(const T& vec, const std::string seperator = "", bool tail = false) {
        // r values can't be passed by reference
        return Cont_PtrJoin<T>(vec, seperator, tail);
    }

    // =================================================================================================================================
    // Shorthand for std::find()
    // picture: ac::has "item" in "vector"
    template<typename A, typename V>
    inline auto Has(const V& item, const A& vec) {
        return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
    }
    template<typename A, typename V>
    inline auto Has(const V& item, const A&& vec) {
        return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
    }
    template<typename A>
    inline auto Has(char const* item, const A& vec) {
        return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
    };
    template<typename A>
    inline auto Has(char const* item, const A&& vec) {
        return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
    };

    // =================================================================================================================================
    // regex match version of std::find()
    template<typename A, typename V>
    bool Cont_MatchOne(const V& in_pattern, const A& vec) {
        std::regex pattern(in_pattern);
        for (typename A::const_iterator iter = vec.begin(); iter != vec.end(); iter++) {
            if (std::regex_match(*iter, pattern)) {
                return true;
            }
        }
        return false;
    }
    template<typename A, typename V>
    inline bool MatchOne(const V& in_pattern, const A& vec) {
        return Cont_MatchOne(in_pattern, vec);
    }
    template<typename A, typename V>
    inline bool MatchOne(const V& in_pattern, const A&& vec) {
        return Cont_MatchOne(in_pattern, vec);
    }
    // ---------------------------------------------------------------------------------------------------------------------------------

    template<typename A, typename V>
    bool Cont_MatchAll(const V& in_pattern, const A& vec) {
        std::regex pattern(in_pattern);
        for (typename A::const_iterator iter = vec.begin(); iter != vec.end(); iter++) {
            if (!std::regex_match(*iter, pattern)) {
                return false;
            }
        }
        return true;
    }
    template<typename A, typename V>
    inline bool MatchAll(const V& in_pattern, const A& vec) {
        return Cont_MatchAll(in_pattern, vec);
    }
    template<typename A, typename V>
    inline bool MatchAll(const V& in_pattern, const A&& vec) {
        return Cont_MatchAll(in_pattern, vec);
    }
    // =================================================================================================================================
    // ret true if regex_search finds a match

    template<typename A, typename V>
    bool Cont_ScanOne(const V& in_pattern, const A& content) {
        std::regex pattern(in_pattern);
        for (typename A::const_iterator iter = content.begin(); iter != content.end(); iter++) {
            if (std::regex_search(*iter, pattern)) {
                return true;
            }
        }
        return false;
    }
    template<typename A, typename V>
    inline bool ScanOne(const V& in_pattern, const A& vec) {
        return Cont_ScanOne(in_pattern, vec);
    }
    template<typename A, typename V>
    inline bool ScanOne(const V& in_pattern, const A&& vec) {
        return Cont_ScanOne(in_pattern, vec);
    }
    // ---------------------------------------------------------------------------------------------------------------------------------

    template<typename A, typename V>
    bool cont_scan_all(const V& in_pattern, const A& content) {
        std::regex pattern(in_pattern);
        for (typename A::const_iterator iter = content.begin(); iter != content.end(); iter++) {
            if (!std::regex_search(*iter, pattern)) {
                return false;
            }
        }
        return true;
    }
    template<typename A, typename V>
    inline bool Scanall(const V& in_pattern, const A& vec) {
        return cont_scan_all(in_pattern, vec);
    }
    template<typename A, typename V>
    inline bool Scanall(const V& in_pattern, const A&& vec) {
        return cont_scan_all(in_pattern, vec);
    }
    // =================================================================================================================================

    template<typename A, typename V>
    V Cont_GetMatches(const A& in_pattern, const V& vec) {
        std::regex pattern(in_pattern);
        V ret_patterns;
        typename V::const_iterator iter;
        for (iter = vec.begin(); iter != vec.end(); ++iter)
        {
            if (std::regex_match(*iter, pattern))
                ret_patterns.push_back(*iter);
        }
        return ret_patterns;
    }
    template<typename A, typename V>
    inline A GetMatches(const V& in_pattern, const A& vec) {
        return Cont_GetMatches(in_pattern, vec);
    }
    template<typename A, typename V>
    inline A GetMatches(const V& in_pattern, const A&& vec) {
        return Cont_GetMatches(in_pattern, vec);
    }
    // =================================================================================================================================
    // return item from vector if a regex segment is found

    template<typename A, typename V>
    V Cont_GetScans(const A& in_pattern, const V& vec) {
        std::regex pattern(in_pattern);
        V ret_patterns;
        typename V::const_iterator iter;
        for (iter = vec.begin(); iter != vec.end(); ++iter)
        {
            if (std::regex_search(*iter, pattern)) // ret whole item so re.h for findall not needed
                ret_patterns.push_back(*iter);
        }
        if (ret_patterns.size()) {
            return ret_patterns;
        } else {
            ret_patterns.resize(1);
            return ret_patterns;
        }
    }
    template<typename A, typename V>
    inline A GetScans(const V& in_pattern, const A& vec) {
        return Cont_GetScans(in_pattern, vec);
    }
    template<typename A, typename V>
    inline A GetScans(const V& in_pattern, const A&& vec) {
        return Cont_GetScans(in_pattern, vec);
    }

    // =================================================================================================================================
    template<typename T>
    inline std::vector<T> Range(const T low, const T high) {
        std::vector<T> vec(high);
        for (T loc = 0; loc < high; loc++) {
            vec[loc] = loc;
        }
        return vec;
    }
    // =================================================================================================================================

    template<typename T = std::string>
    inline std::string Ditto(const T item, const size_t repeate_count, const std::string seperator = "", bool tail = false) {
        std::string ret_str;
        for (size_t count = 0; count < repeate_count; count++) {
            ret_str += item + seperator;
        }
        if (tail == false && seperator.size()) {
            ret_str = ret_str.substr(0, ret_str.size() - seperator.size());
        }
        return ret_str;
    }

    // =================================================================================================================================

    // ac_vice was not intended to be used directly
    // for clarity, and usability, it is suggested to use slice/dice which call ac_vice
    // slice is like python's slice and dice is the same but removes what slice skips with 'z'
    template<typename T>
    T Vice(T i_arr, double x, double y, double z, char removal_method) {
        size_t m_size = i_arr.size();
        T n_arr;
        n_arr.reserve(m_size + 4);

        double n_arr_size = static_cast<double>(m_size - 1);

        if (z >= 0) {

            if (x < 0) { x += n_arr_size; }

            if (!y) { y = n_arr_size; } else if (y < 0) { y += n_arr_size; }
            ++y;

            if (x > y) { return n_arr; }

            typename T::iterator iter = i_arr.begin();
            typename T::iterator stop = i_arr.begin() + y;

            if (z == 0) { // forward direction with no skipping
                for (iter += x; iter != stop; ++iter)
                    n_arr.push_back(*iter);
            } else if (removal_method == 's') { // forward direction with skipping
                double iter_insert = 0;
                --z;
                for (iter += x; iter != stop; ++iter) {
                    if (!iter_insert) {
                        n_arr.push_back(*iter);
                        iter_insert = z;
                    } else {
                        --iter_insert;
                    }
                }
            } else {
                double iter_insert = 0;
                --z;
                for (iter += x; iter != stop; ++iter) {
                    if (!iter_insert) {
                        iter_insert = z;
                    } else {
                        n_arr.push_back(*iter);
                        --iter_insert;
                    }
                }
            }
        } else { // reverse direction
            z = static_cast<size_t>(z = z * -1 - 1);
            if (!x) { x = n_arr_size; } else if (x < 0) { x += n_arr_size; }

            if (!y) { y = 0; } else if (y < 0) { y += n_arr_size; }

            if (y > x) { return n_arr; }

            x = static_cast<size_t>(x);
            y = static_cast<size_t>(y);

            typename T::reverse_iterator iter = i_arr.rend() - x - 1;
            typename T::reverse_iterator stop = i_arr.rend() - y;

            size_t iter_insert = 0;

            if (z == 0) {
                for (; iter != stop; ++iter) {
                    if (!iter_insert)
                        n_arr.push_back(*iter);
                }
            } else if (removal_method == 's') {
                for (; iter != stop; ++iter) {
                    if (!iter_insert) {
                        n_arr.push_back(*iter);
                        iter_insert = z;
                    } else {
                        --iter_insert;
                    }
                }
            } else {
                for (; iter != stop; ++iter) {
                    if (!iter_insert) {
                        iter_insert = z;
                    } else {
                        n_arr.push_back(*iter);
                        --iter_insert;
                    }
                }
            }
        }
        return n_arr;
    }

    template<typename T>
    T Slice(T i_arr, double x = 0, double y = 0, double z = 0) {
        return Vice(i_arr, x, y, z, 's');
    }

    template<typename T>
    T Dice(T i_arr, double x = 0, double y = 0, double z = 0) {
        return Vice(i_arr, x, y, z, 'd');
    }
}
