#pragma once

#include<map>
#include<unordered_map>
#include<vector>
#include<string>
#include<algorithm>
#include<regex>
#include<sstream>


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


namespace ord // ORDer
{
	
	// =================================================================================================================================

	template<typename A, typename B>
	std::vector<A> keys(const std::map<A, B>& xmap);

	template<typename A, typename B>
	std::vector<A> keys(const std::unordered_map<A, B>& xmap);

	template<typename T = std::string>
	bool in(const T& item, const std::vector<T>& vec);

	template<typename T = std::string>
	bool has(const std::vector<T>& vec, const T& item);

	// =================================================================================================================================
	// join vector items to a string
	template<typename T>
	std::string xjoin(const T& vec, const std::string seperator = " ", bool tail = false) {
		std::ostringstream ostr;
		for (auto& i : vec)
			ostr << i << seperator;
		if (tail == false)
			return (ostr.str().substr(0, ostr.str().size() - seperator.size()));
		return ostr.str();
	}
	template<typename T>
	std::string join(const T& vec, const std::string seperator = " ", bool tail = false) {
		return xjoin<T>(vec, seperator, tail);
	}
	template<typename T>
	std::string join(const T&& vec, const std::string seperator = " ", bool tail = false) {
		// r values can't be passed by reference
		return xjoin<T>(vec, seperator, tail);
	}
	// =================================================================================================================================
	// map1 = {key1 , var1}
	// map2 = {var1,  var2}
	// output
	// map1 = {key1,   var1, "new_keyname", var2}

	// map1, map2, key1, new_keyname

	template<typename X, typename Y>
	inline void relational_copy(std::map<X, Y>& map1, const std::string new_keyname, \
		const std::map<X, Y>& map2, const std::string key1) {
		if (has(keys(map2), map1.at(key1))) {
			map1.insert({ {new_keyname, map2.at(map1.at(key1))} });
		}
	}
	template<typename X, typename Y>
	inline void relational_copy(std::unordered_map<X, Y>& map1, const std::string new_keyname, \
		const std::unordered_map<X, Y>& map2, const std::string key1) {
		if (has(keys(map2), map1.at(key1))) {
			map1.insert({ {new_keyname, map2.at(map1.at(key1))} });
		}
	}
	template<typename X, typename Y>
	inline void relational_copy(std::map<X, Y>& map1, const std::string new_keyname, \
		const std::unordered_map<X, Y>& map2, const std::string key1) {
		if (has(keys(map2), map1.at(key1))) {
			map1.insert({ {new_keyname, map2.at(map1.at(key1))} });
		}
	}
	template<typename X, typename Y>
	inline void relational_copy(std::unordered_map<X, Y>& map1, const std::string new_keyname, \
		const std::map<X, Y>& map2, const std::string key1) {
		if (has(keys(map2), map1.at(key1))) {
			map1.insert({ {new_keyname, map2.at(map1.at(key1))} });
		}
	}
	// =================================================================================================================================
	// find Key > does it exist?

	template<typename X, typename Y>
	inline bool xhas_key(const std::map<X, Y>& d_map, const std::string& key) {
		if (bool((d_map.find(key) != d_map.end()))) {
			if (d_map.at(key).size() > 0) {
				return true;
			}
			return false;
		}
		return false;
	}

	template<typename X, typename Y>
	inline bool xhas_key(const std::unordered_map<X, Y>& d_map, const std::string key) {
		if (bool((d_map.find(key) != d_map.end()))) {
			if (d_map.at(key).size() > 0) {
				return true;
			}
			return false;
		}
		return false;
	}

	template<typename X, typename Y>
	inline bool has_key(const std::unordered_map<X, Y>& d_map, const std::string key) {
		return xhas_key(d_map, key);
	}
	template<typename X, typename Y>
	inline bool has_key(const std::unordered_map<X, Y>&& d_map, const std::string key) {
		return xhas_key(d_map, key);
	}
	template<typename X, typename Y>
	inline bool has_key(const std::map<X, Y>& d_map, const std::string key) {
		return xhas_key(d_map, key);
	}
	template<typename X, typename Y>
	inline bool has_key(const std::map<X, Y>&& d_map, const std::string key) {
		return xhas_key(d_map, key);
	}

	template<typename X, typename Y>
	inline bool in_key(const std::string key, const std::unordered_map<X, Y>& d_map) {
		return xhas_key(d_map, key);
	}
	template<typename X, typename Y>
	inline bool in_key(const std::string key, const std::unordered_map<X, Y>&& d_map) {
		return xhas_key(d_map, key);
	}
	template<typename X, typename Y>
	inline bool in_key(const std::string key, const std::map<X, Y>& d_map) {
		return xhas_key(d_map, key);
	}
	template<typename X, typename Y>
	inline bool in_key(const std::string key, const std::map<X, Y>&& d_map) {
		return xhas_key(d_map, key);
	}


	// =================================================================================================================================
	// ret vector of all keys from map
	template<typename A, typename B>
	inline std::vector<A> keys(const std::map<A, B>& xmap) {
		std::vector<A> vec;
		for (typename std::map<A, B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
			vec.push_back(iter->first);
		return vec;
	}
	template<typename A, typename B>
	inline std::vector<A> keys(const std::map<A, B>&& xmap) {
		std::vector<A> vec;
		for (typename std::map<A, B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
			vec.push_back(iter->first);
		return vec;
	}        // ret vector of all values from map
	template<typename A, typename B>
	inline std::vector<A> keys(const std::unordered_map<A, B>& xmap) {
		std::vector<A> vec;
		for (typename std::unordered_map<A, B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
			vec.push_back(iter->first);
		return vec;
	}
	template<typename A, typename B>
	inline std::vector<A> keys(const std::unordered_map<A, B>&& xmap) {
		std::vector<A> vec;
		for (typename std::unordered_map<A, B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
			vec.push_back(iter->first);
		return vec;
	}        // ret vector of all values from map
	template<typename A, typename B>
	inline std::vector<B> key_values(const std::map<A, B>& xmap) {
		std::vector<B> vec;
		for (typename std::unordered_map<A, B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
			vec.push_back(iter->second);
		return vec;
	}
	template<typename A, typename B>
	inline std::vector<B> key_values(const std::map<A, B>&& xmap) {
		std::vector<B> vec;
		for (typename std::unordered_map<A, B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
			vec.push_back(iter->second);
		return vec;
	}
	template<typename A, typename B>
	inline std::vector<B> key_values(const std::unordered_map<A, B>& xmap) {
		std::vector<B> vec;
		for (typename std::unordered_map<A, B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
			vec.push_back(iter->second);
		return vec;
	}
	template<typename A, typename B>
	inline std::vector<B> key_values(const std::unordered_map<A, B>&& xmap) {
		std::vector<B> vec;
		for (typename std::unordered_map<A, B>::const_iterator iter = xmap.begin(); iter != xmap.end(); ++iter)
			vec.push_back(iter->second);
		return vec;
	}
	// =================================================================================================================================
	// Shorthand for std::find()
	template<typename T>
	inline bool has(const std::vector<T>& vec, const T& item) {
		return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
	}
	template<typename T>
	inline bool has(const std::vector<T>&& vec, const T& item) {
		return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
	}
	inline bool has(const std::vector<std::string>& vec, char const* item) {
		return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
	};
	inline bool has(const std::vector<std::string>&& vec, char const* item) {
		return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
	};

	// 'in' just reverses the order of input as 'has' (they are both the same thing)
	template<typename T>
	inline bool in(const T& item, const std::vector<T>& vec) {
		return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
	}
	template<typename T>
	inline bool in(const T& item, const std::vector<T>&& vec) {
		return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
	}
	inline bool in(char const* item, const std::vector<std::string>& vec) {
		return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
	};
	inline bool in(char const* item, const std::vector<std::string>&& vec) {
		return (bool(std::find(vec.begin(), vec.end(), item) != vec.end()));
	};


	// =================================================================================================================================
	// regex match version of std::find()
	bool xmatch_one(const std::string& in_pattern, const std::vector<std::string>& vec);

	inline bool match_one(const std::string& in_pattern, const std::vector<std::string>& vec) {
		return xmatch_one(in_pattern, vec);
	}
	inline bool match_one(const std::string in_pattern, const std::vector<std::string>&& vec) {
		return xmatch_one(in_pattern, vec);
	}
	bool xmatch_all(const std::string& in_pattern, const std::vector<std::string>& vec);

	inline bool match_all(const std::string& in_pattern, const std::vector<std::string>& vec) {
		return xmatch_all(in_pattern, vec);
	}
	inline bool match_all(const std::string in_pattern, const std::vector<std::string>&& vec) {
		return xmatch_all(in_pattern, vec);
	}
	// =================================================================================================================================
	// ret true if regex_search finds a match

	bool xscan_one(const std::string& in_pattern, const std::vector<std::string>& vec);

	inline bool scan_one(const std::string& in_pattern, const std::vector<std::string>& vec) {
		return xscan_one(in_pattern, vec);
	}
	inline bool scan_one(const std::string& in_pattern, const std::vector<std::string>&& vec) {
		return xscan_one(in_pattern, vec);
	}
	bool xscan_all(const std::string& in_pattern, const std::vector<std::string>& vec);

	inline bool scan_all(const std::string& in_pattern, const std::vector<std::string>& vec) {
		return xscan_all(in_pattern, vec);
	}
	inline bool scan_all(const std::string& in_pattern, const std::vector<std::string>&& vec) {
		return xscan_all(in_pattern, vec);
	}
	// =================================================================================================================================
	std::vector<std::string> xret_matches(const std::string& in_pattern, const std::vector<std::string>& vec);

	inline std::vector<std::string> ret_matches(const std::string& in_pattern, const std::vector<std::string>& vec) {
		return xret_matches(in_pattern, vec);
	}
	inline std::vector<std::string> ret_matches(const std::string in_pattern, const std::vector<std::string>&& vec) {
		return xret_matches(in_pattern, vec);
	}
	// =================================================================================================================================
	// return item from vector if a regex segment is found

	std::vector<std::string> xret_scans(const std::string& in_pattern, const std::vector<std::string>& vec);

	inline std::vector<std::string> ret_scans(const std::string& in_pattern, const std::vector<std::string>& vec) {
		return xret_scans(in_pattern, vec);
	}
	inline std::vector<std::string> ret_scans(const std::string& in_pattern, const std::vector<std::string>&& vec) {
		return xret_scans(in_pattern, vec);
	}
	template<typename T = int>
	inline std::vector<T> range(const T low, const T high) {
		std::vector<T> vec(high);
		for (T loc = 0; loc < high; loc++) {
			vec[loc] = loc;
		}
		return vec;
	}
	// =================================================================================================================================

	template<typename T = std::string>
	inline std::string ditto(const T item, const size_t repeate_count, const std::string seperator = "", bool tail = false) {
		std::string ret_str;
		for (size_t count = 0; count < repeate_count; count++) {
			ret_str += item + seperator;
		}
		if (tail == false && seperator.size()) {
			ret_str = ret_str.substr(0, ret_str.size() - seperator.size());
		}
		return ret_str;
	}

	template<typename T>
	T slice(T i_arr, double x = 0, double y = 0, double z = 0) {

		size_t m_size = i_arr.size();
		T n_arr;
		n_arr.reserve(m_size);

		double n_arr_size = static_cast<double>(m_size - 1);

		if (z >= 0) {

			if (x < 0) { x += n_arr_size; }

			if (!y) { y = n_arr_size; }
			else if (y < 0) { y += n_arr_size; }
			++y;

			if (x > y) { return n_arr; }

			typename T::iterator iter = i_arr.begin();
			typename T::iterator stop = i_arr.begin() + y;

			if (z == 0) { // forward direction with no skipping
				for (iter += x; iter != stop; ++iter)
					n_arr.push_back(*iter);
			}
			else { // forward direction with skipping
				double iter_insert = 0;
				--z;
				for (iter += x; iter != stop; ++iter) {
					if (!iter_insert) {
						n_arr.push_back(*iter);
						iter_insert = z;
					}
					else {
						--iter_insert;
					}
				}
			}
		}
		else { // reverse direction
			z = static_cast<size_t>(z = z * -1 - 1);
			if (!x) { x = n_arr_size; }
			else if (x < 0) { x += n_arr_size; }

			if (!y) { y = 0; }
			else if (y < 0) { y += n_arr_size; }

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
			}
			else {
				for (; iter != stop; ++iter) {
					if (!iter_insert) {
						n_arr.push_back(*iter);
						iter_insert = z;
					}
					else {
						--iter_insert;
					}
				}
			}
		}
		return n_arr;
	}
}
