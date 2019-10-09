﻿#pragma once
#pragma warning (disable : 26444) // allow anynomous objects

/*
* Copyright[2019][Joel Leagues aka Scourge]
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

#include<vector>
#include<initializer_list>
#include<string>
#include<regex>
#include<sstream>
#include<set>


#if (defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32))
	using size64_t = __int64;
#else
	using size64_t = __int64_t;
#endif

// THIS TEMPLATE WAS TESTED WITH xstring BY RADICALWARE.NET

template<typename T>
class xvector : public std::vector<T>
{
public:
	inline ~xvector();
	inline xvector() {};
	inline xvector(size_t sz)                    : std::vector<T>(sz) {};
	inline xvector(const std::vector<T>& vec)    : std::vector<T>(vec) {};
	inline xvector(std::vector<T>&& vec)         : std::vector<T>(vec) {};
	inline xvector(std::initializer_list<T> lst) : std::vector<T>(lst) {};
	inline void operator=(const xvector<T>& other);

	template<typename P = T> 
	inline bool has(const P* item);
	inline bool has(const T& item);
	inline bool has(T&& item);
	inline bool has(char const* item);

	inline bool lacks(const T& item);
	inline bool lacks(T&& item);
	inline bool lacks(char const* item);

	// not const because sort is used (this only changes order, it does not remove dups)
	template<typename L = xvector<T>>
	inline xvector<T> common(L & item);
	template<typename L = xvector<T>>
	inline xvector<T> common(L && item);
	inline xvector<T> common(char const* item);

	inline void operator<<(const T& item);
	inline void operator<<(const T&& item);

	inline void operator*=(const size_t count);

	inline void add();
	inline void add(const T& val); 
	template <typename First, typename... Rest>
	inline void add(const First& first, const Rest& ... rest);

	inline void add_char_strings(int strC, char** strV);

	template<typename O>
	inline bool operator>(const O& other);
	template<typename O>
	inline bool operator<(const O& other);
	template<typename O>
	inline bool operator==(const O& other);
	template<typename O>
	inline bool operator!=(const O& other);

	inline bool operator>(const size_t value);
	inline bool operator<(const size_t value);
	inline bool operator==(const size_t value);
	inline bool operator!=(const size_t value);

	T back(size_t value = 1) const;
	T* back_ptr(size_t value = 1);

	template<typename C>
	inline xvector<C> convert();

	template<typename C, typename F>
	inline xvector<C> convert(F function);

	template<typename N = unsigned int>
	inline xvector<xvector<T>> split(N count = 1);

	inline void operator+=(const xvector<T>& other);
	inline xvector<T> operator+(const xvector<T>& other);

	inline void organize();
	inline void remove_dups();

	template<typename F>
	inline void sort(F func);

	inline xvector<T*> ptrs();

	template<typename F>
	inline void proc(const F& function);
	template<typename V, typename F>
	inline void proc(V& value, const F& function);

	template<typename V = T, typename F>
	inline xvector<V> render(const F& function);
	template<typename V = T, typename F>
	inline xvector<V> render(V& value, const F& function);
	template<typename V = T, typename F>
	inline xvector<V> render(V&& value, const F& function);

	template<typename S>
	inline S sjoin(const S& seperator = "\0", bool tail = false);

	// =================================== DESIGNED FOR NUMERIC BASED VECTORS ===================================

	inline size_t sum();

	// =================================== DESIGNED FOR STRING  BASED VECTORS ===================================

	inline T join(const T& str = "");
	inline T join(const char str);
	inline T join(const char* str);

	inline bool match_one(const T& in_pattern) const;
	inline bool match_one(T&& in_pattern) const;
	inline bool match_one(char const* in_pattern) const;

	inline bool match_all(const T& in_pattern) const;
	inline bool match_all(T&& in_pattern) const;
	inline bool match_all(char const* in_pattern) const;

	inline bool scan_one(const T& in_pattern) const;
	inline bool scan_one(T&& in_pattern) const;
	inline bool scan_one(char const* in_pattern) const;

	inline bool scan_all(const T& in_pattern) const;
	inline bool scan_all(T&& in_pattern) const;
	inline bool scan_all(char const* in_pattern) const;
	
	inline xvector<T> take(const T& in_pattern) const;
	inline xvector<T> take(T&& in_pattern) const;
	inline xvector<T> take(char const* in_pattern) const;

	inline xvector<T> remove(const T& in_pattern) const;
	inline xvector<T> remove(T&& in_pattern) const;
	inline xvector<T> remove(char const* in_pattern) const;

	inline xvector<T> sub_all(const T& in_pattern, const T& replacement) const;
	inline xvector<T> sub_all(T&& in_pattern, T&& replacement) const;
	inline xvector<T> sub_all(char const* in_pattern, char const* replacement) const;
	
	inline T ditto(const size_t repeate_count, const T& seperator = "", bool tail = false);
	inline T ditto(const size_t repeate_count, T&& seperator = "", bool tail = false);
	inline T ditto(const size_t repeate_count, char const* seperator = "", bool tail = false);

	// double was chose to hold long signed and unsigned values
	inline xvector<T> operator()(double x = 0, double y = 0, double z = 0, const char removal_method = 's') const;
	// s = slice perserves values you land on 
	// d = dice  removes values you land on
	// s/d only makes a difference if you modify the 'z' value

	// =================================== DESIGNED FOR STRING BASED VECTORS ===================================

};
// =============================================================================================================

template<typename T>
inline xvector<T>::~xvector()
{

}

template<typename T>
void xvector<T>::operator=(const xvector<T>& other) {
	this->clear();
	this->reserve(other.size());
	this->insert(this->begin(), other.begin(), other.end());
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename P>
inline bool xvector<T>::has(const P* item)
{
	for (typename xvector<T>::iterator it = this->begin(); it != this->end(); it++) {
		if (**it == *item)
			return true;
	}
	return false;
}

template<typename T>
bool xvector<T>::has(const T& item) {
	return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool xvector<T>::has(T&& item) {
	return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool xvector<T>::has(char const* item) {
	return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
bool xvector<T>::lacks(T&& item) {
	return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool xvector<T>::lacks(const T& item) {
	return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool xvector<T>::lacks(char const* item) {
	return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename L>
xvector<T> xvector<T>::common(L& item) {
	std::sort(this->begin(), this->end());
	std::sort(item.begin(), item.end());

	xvector<T> vret(this->size() + item.size());
	set_intersection(this->begin(), this->end(), item.begin(), item.end(), vret.begin());
	return vret;
}

template<typename T>
template<typename L>
xvector<T> xvector<T>::common(L&& item) {
	return this->common(item);
}

template<typename T>
xvector<T> xvector<T>::common(char const* item) {

	size_t size = strlen(item);
	xvector<char> c_vec(size);

	for (int i = 0; i < size; i++)
		c_vec << item[i];

	return this->common(c_vec);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline void xvector<T>::operator<<(const T& item)
{
	this->emplace_back(item);
}

template<typename T>
inline void xvector<T>::operator<<(const T&& item)
{
	this->emplace_back(item);
}

template<typename T>
inline void xvector<T>::operator*=(const size_t count)
{
	xvector<T>* tmp = new xvector<T>;
	tmp->reserve(this->size() + 1);
	*tmp = *this;
	
	this->reserve(this->size() * count + 1);
	for (int i = 0; i < count - 1; i++)
		this->insert(this->end(), tmp->begin(), tmp->end());
	delete tmp;
}

template<typename T>
inline void xvector<T>::add() {}
template<typename T>
inline void xvector<T>::add(const T& val)
{
	*this << val;
}
template<typename T>
template<typename First, typename ...Rest>
inline void xvector<T>::add(const First& first, const Rest& ...rest)
{
	*this << first;
	this->add(rest...);
}

template<typename T>
inline void xvector<T>::add_char_strings(int strC, char** strV)
{
	for (int i = 0; i < strC; i++)
		this->push_back(T(strV[i]));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename O>
inline bool xvector<T>::operator>(const O& other)
{
	return this->size() > other.size();
}

template<typename T>
template<typename O>
inline bool xvector<T>::operator<(const O& other)
{
	return this->size() < other.size();
}

template<typename T>
template<typename O>
inline bool xvector<T>::operator==(const O& other)
{
	for (T& it : other) {
		if (this->lacks(it))
			return false;
	}
	return true;
}

template<typename T>
template<typename O>
inline bool xvector<T>::operator!=(const O& other)
{
	for (T& it : other) {
		if (this->lacks(it))
			return true;
	}
	return false;
}
// --------------------------------------------------------
template<typename T>
inline bool xvector<T>::operator>(const size_t value)
{
	return this->size() > value;
}

template<typename T>
inline bool xvector<T>::operator<(const size_t value)
{
	return this->size() < value;
}

template<typename T>
inline bool xvector<T>::operator==(const size_t value)
{
	return this->size() == value;
}

template<typename T>
inline bool xvector<T>::operator!=(const size_t value)
{
	return this->size() != value;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline T xvector<T>::back(size_t value) const
{
	return this->operator[](this->size() - value);
}

template<typename T>
inline T* xvector<T>::back_ptr(size_t value)
{
	return &this->operator[](this->size() - value);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename C>
inline xvector<C> xvector<T>::convert()
{
	xvector<C> ret;
	for (typename xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
		ret << C(*it);
	return ret;
}


template<typename T>
template<typename C, typename F>
inline xvector<C> xvector<T>::convert(F function)
{
	xvector<C> ret;
	for (typename xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
		ret << function(*it);
	return ret;
}

template<typename T>
template<typename N>
inline xvector<xvector<T>> xvector<T>::split(N count)
{
	if (count < 2)
		return xvector<xvector<T>>{ *this };

	xvector<xvector<T>> ret_vec;
	ret_vec.reserve(static_cast<size_t>(count) + 1);
	if (!this->size())
		return ret_vec;

	N reset = count;
	count = 0;
	const N new_size = static_cast<N>(this->size()) / reset;
	for (typename xvector<T>::iterator it = this->begin(); it != this->end(); it++) {
		if (count == 0) {
			count = reset;
			ret_vec.push_back(xvector<T>({ *it })); // create new xvec and add first el
			ret_vec[ret_vec.size() - 1].reserve(static_cast<size64_t>(new_size));
		}
		else {
			ret_vec[ret_vec.size() - 1] << *it;
		}
		count--;
	}
	return ret_vec;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
void xvector<T>::operator+=(const xvector<T>& other)
{
	this->insert(this->end(), other.begin(), other.end());
}

template<typename T>
xvector<T> xvector<T>::operator+(const xvector<T>& other) {
	size_t sz = this->size();
	*this += other;
	xvector<T> ret_val = *this;
	this->resize(sz);
	return ret_val;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline void xvector<T>::organize()
{
	std::multiset<T> set_arr;
	for (typename xvector<T>::iterator it = this->begin(); it != this->end(); it++)
		set_arr.insert(*it);

	this->clear();
	this->reserve(set_arr.size());

	for (typename std::multiset<T>::iterator it = set_arr.begin(); it != set_arr.end(); it++)
		this->push_back(*it);
}

template<typename T>
inline void xvector<T>::remove_dups()
{
	std::set<T> set_arr;
	for (typename xvector<T>::iterator it = this->begin(); it != this->end(); it++)
		set_arr.insert(*it);

	this->clear();
	this->reserve(set_arr.size());

	for (typename std::set<T>::iterator it = set_arr.begin(); it != set_arr.end(); it++)
		this->push_back(*it);
}

// -----------------------------------------------------------------------------------------------

template<typename T>
template<typename F>
inline void xvector<T>::sort(F func)
{
	std::sort(this->begin(), this->end(), func);
}

template<typename T>
inline xvector<T*> xvector<T>::ptrs()
{
	xvector<T*> ret_vec;
	for (T& item : *this) {
		ret_vec << &item;
	}
	return ret_vec;
}

template<typename T>
template<typename F>
inline void xvector<T>::proc(const F& function)
{
	for (size_t i = 0; i < this->size(); i++)
		function(this->operator[](i));
}

template<typename T>
template<typename V, typename F>
inline void xvector<T>::proc(V& value, const F& function)
{
	for (typename xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
		function(value, *it);
}


template<typename T>
template<typename V, typename F>
inline xvector<V> xvector<T>::render(const F& function)
{
	xvector<V> vret;
	for (typename xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
		vret.push_back(function(*it));
	return vret;
}

template<typename T>
template<typename V, typename F>
inline xvector<V> xvector<T>::render(V& value, const F& function)
{
	xvector<V> vret;
	for (typename xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
		vret.push_back(function(value, *it));
	return vret;
}

template<typename T>
template<typename V, typename F>
inline xvector<V> xvector<T>::render(V&& value, const F& function)
{
	xvector<V> vret;
	for (typename xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
		vret.push_back(function(value, *it));
	return vret;
}

template<typename T>
template<typename S>
S xvector<T>::sjoin(const S& seperator, bool tail) {
	std::ostringstream ostr;
	if (seperator[0] != '\0') {
		for (auto& i : *this) {
			ostr << i;
			ostr << seperator;
		}
		if (tail == false)
			return (ostr.str().substr(0, ostr.str().size() - seperator.size()));
	}
	else {
		for (auto& i : *this)
			ostr << i;
	}

	return ostr.str();
}


// =============================================================================================================


template<typename T>
inline size_t xvector<T>::sum()
{
	size_t num = 0;
	for (typename xvector<T>::iterator it = this->begin(); it != this->end(); it++) {
		num += *it;
	}
	return num;
}

// =============================================================================================================

template<typename T>
inline T xvector<T>::join(const T& str)
{
	T ret;
	for (typename xvector<T>::iterator it = this->begin(); it != this->end(); it++)
		ret += *it + str;

	return ret.substr(0, ret.length() - str.size());
}

template<typename T>
T xvector<T>::join(const char str)
{
	T ret;
	for (typename xvector<T>::iterator it = this->begin(); it != this->end(); it++)
		ret += *it + str;

	return ret.substr(0, ret.length() - 1);
}

template<typename T>
T xvector<T>::join(const char* str)
{
	T ret;
	for (typename xvector<T>::iterator it = this->begin(); it != this->end(); it++)
		ret += *it + str;

	return ret.substr(0, ret.length() - strlen(str));
}

template<typename T>
bool xvector<T>::match_one(const T& in_pattern) const {
	std::regex pattern(in_pattern.c_str());
	for (typename xvector<T>::const_iterator iter = this->begin(); iter != this->end(); iter++) {
		if (std::regex_match(*iter, pattern)) {
			return true;
		}
	}
	return false;
}

template<typename T>
bool xvector<T>::match_one(T&& in_pattern) const {
	return this->match_one(in_pattern);
}

template<typename T>
bool xvector<T>::match_one(char const* in_pattern) const {
	return this->match_one(T(in_pattern));
}

// =============================================================================================================

template<typename T>
bool xvector<T>::match_all(const T& in_pattern) const {
	std::regex pattern(in_pattern);
	for (typename T::const_iterator iter = this->begin(); iter != this->end(); iter++) {
		if (!std::regex_match(*iter, pattern)) {
			return false;
		}
	}
	return true;
}

template<typename T>
bool xvector<T>::match_all(T&& in_pattern) const {
	return this->match_all(in_pattern);
}

template<typename T>
bool xvector<T>::match_all(char const* in_pattern) const {
	return this->match_all(T(in_pattern));
}

// =============================================================================================================


template<typename T>
bool xvector<T>::scan_one(const T& in_pattern) const {
	std::regex pattern(in_pattern);
	for (typename xvector<T>::const_iterator iter = this->begin(); iter != this->end(); iter++) {
		if (std::regex_search(*iter, pattern)) {
			return true;
		}
	}
	return false;
}

template<typename T>
bool xvector<T>::scan_one(T&& in_pattern) const {
	return this->scan_one(in_pattern);
}

template<typename T>
bool xvector<T>::scan_one(char const* in_pattern) const {
	return this->scan_one(T(in_pattern));
}

// =============================================================================================================

template<typename T>
bool xvector<T>::scan_all(const T& in_pattern) const {
	std::regex pattern(in_pattern);
	for (typename T::const_iterator iter = this->begin(); iter != this->end(); iter++) {
		if (!std::regex_search(*iter, pattern)) {
			return false;
		}
	}
	return true;
}

template<typename T>
bool xvector<T>::scan_all(T&& in_pattern) const {
	return this->scan_all(in_pattern);
}

template<typename T>
bool xvector<T>::scan_all(char const* in_pattern) const {
	return this->scan_all(T(in_pattern));
}
// =============================================================================================================

template<typename T>
inline xvector<T> xvector<T>::take(const T& in_pattern) const
{
	xvector<T> ret_vec;
	std::regex pattern(in_pattern);
	ret_vec.reserve(this->size()+1);
	for (size_t i = 0; i < this->size(); i++) {
		if ((std::regex_search((*this)[i], pattern)))
			ret_vec.push_back((*this)[i]);
	}
	return ret_vec;
}

template<typename T>
inline xvector<T> xvector<T>::take(T&& in_pattern) const
{
	return this->take(in_pattern);
}

template<typename T>
inline xvector<T> xvector<T>::take(char const* in_pattern) const
{
	return this->take(T(in_pattern));
}


template<typename T>
inline xvector<T> xvector<T>::remove(const T& in_pattern) const
{
	xvector<T> ret_vec;
	ret_vec.reserve(this->size()+1);
	std::regex pattern(in_pattern.c_str());
	for (size_t i = 0; i < this->size(); i++) {
		if(!(std::regex_search((*this)[i].c_str(), pattern)))
			ret_vec.push_back((*this)[i]);
	}
	return ret_vec;
}

template<typename T>
inline xvector<T> xvector<T>::remove(T&& in_pattern) const
{
	return this->remove(in_pattern);
}

template<typename T>
inline xvector<T> xvector<T>::remove(char const* in_pattern) const
{
	return this->remove(T(in_pattern));
}

// =============================================================================================================

template<typename T>
inline xvector<T> xvector<T>::sub_all(const T& in_pattern, const T& replacement) const
{
	xvector<T> ret_vec;
	std::regex pattern(in_pattern.c_str());
	ret_vec.reserve(this->size()+1);
	for (typename xvector<T>::const_iterator iter = this->begin(); iter != this->end(); iter++) 
		ret_vec << T(std::regex_replace(*iter, pattern, replacement));
	return ret_vec;
}

template<typename T>
inline xvector<T> xvector<T>::sub_all(T&& in_pattern, T&& replacement) const
{
	return this->sub_all(in_pattern, replacement);
}

template<typename T>
inline xvector<T> xvector<T>::sub_all(char const* in_pattern, char const* replacement) const
{
	return this->sub_all(T(in_pattern), T(replacement));
}

// =============================================================================================================

template<typename T>
inline T xvector<T>::ditto(const size_t repeate_count, const T& seperator, bool tail /* = false */) {
	T ret_str;
	for (size_t count = 0; count < repeate_count; count++)
		ret_str += (*this) + seperator;

	if (tail == false && seperator.size())
		ret_str = ret_str.substr(0, ret_str.size() - seperator.size());

	return ret_str;
}


template<typename T>
inline T xvector<T>::ditto(const size_t repeate_count, T&& seperator, bool tail /* = false */) {
	return this->ditto(repeate_count, seperator, tail);
}

template<typename T>
inline T xvector<T>::ditto(const size_t repeate_count, char const* seperator, bool tail /* = false */) {
	return this->ditto(repeate_count, seperator, tail);
}

// =============================================================================================================

template<typename T>
xvector<T> xvector<T>::operator()(double x, double y, double z, const char removal_method) const {

	size_t m_size = this->size();
	xvector<T> n_arr;
	n_arr.reserve(m_size + 4);

	double n_arr_size = static_cast<double>(m_size - 1);

	if (z >= 0) {

		if (x < 0) { x += n_arr_size; }

		if (!y) { y = n_arr_size; }
		else if (y < 0) { y += n_arr_size; }
		++y;

		if (x > y) { return n_arr; }

		typename xvector<T>::const_iterator iter = this->begin();
		typename xvector<T>::const_iterator stop = this->begin() + static_cast<size_t>(y);

		if (z == 0) { // forward direction with no skipping
			for (iter += static_cast<size_t>(x); iter != stop; ++iter)
				n_arr.push_back(*iter);
		}
		else if (removal_method == 's') { // forward direction with skipping
			double iter_insert = 0;
			--z;
			for (iter += static_cast<size_t>(x); iter != stop; ++iter) {
				if (!iter_insert) {
					n_arr.push_back(*iter);
					iter_insert = z;
				}
				else {
					--iter_insert;
				}
			}
		}
		else {
			double iter_insert = 0;
			--z;
			for (iter += static_cast<size_t>(x); iter != stop; ++iter) {
				if (!iter_insert) {
					iter_insert = z;
				}
				else {
					n_arr.push_back(*iter);
					--iter_insert;
				}
			}
		}
	}
	else { // reverse direction
		z = z * -1 - 1;
		if (!x) { x = n_arr_size; }
		else if (x < 0) { x += n_arr_size; }

		if (!y) { y = 0; }
		else if (y < 0) { y += n_arr_size; }

		if (y > x) { return n_arr; }
		
		typename xvector<T>::const_reverse_iterator iter = this->rend() - static_cast<size_t>(x) - 1;
		typename xvector<T>::const_reverse_iterator stop = this->rend() - static_cast<size_t>(y);

		size_t iter_insert = 0;

		if (z == 0) {
			for (; iter != stop; ++iter) {
				if (!iter_insert)
					n_arr.push_back(*iter);
			}
		}
		else if (removal_method == 's') {
			for (; iter != stop; ++iter) {
				if (!iter_insert) {
					n_arr.push_back(*iter);
					iter_insert = static_cast<size_t>(z);
				}
				else {
					--iter_insert;
				}
			}
		}
		else {
			for (; iter != stop; ++iter) {
				if (!iter_insert) {
					iter_insert = static_cast<size_t>(z);
				}
				else {
					n_arr.push_back(*iter);
					--iter_insert;
				}
			}
		}
	}
	return n_arr;
}
