#pragma once

/*
* Copyright[2019][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, const V*ersion 2.0 (the "License");
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

#include<map>
#include<unordered_map>
#include<initializer_list>
#include<utility>
#include<memory>

#include "xvector.h"
#include "xstring.h"

#include "const2_xmap.h"
#include "const_ptr_xmap.h"
#include "const_val_xmap.h"
#include "ptr_val_xmap.h"
#include "val2_xmap.h"

template<typename K, typename V> class xmap<const K*, const V*>;
template<typename K, typename V> class xmap<const K*,       V*>;
template<typename K, typename V> class xmap<const K*,       V >;
template<typename K, typename V> class xmap<      K*, const V >;
template<typename K, typename V> class xmap;

template<typename K, typename V>
class xmap<K*, V*> : public std::unordered_map<K*, V*>
{
private:
	xvector<const K*>* m_keys = nullptr;   // used for setting orders to your keys 
	xmap<const V*, const K*>* m_rev_map = nullptr;   // go from KVPs to VKPs

public:
	// ======== INITALIZATION ========================================================================
	inline xmap();
	inline ~xmap();

	inline xmap(std::initializer_list<std::pair<K*, V*>> init);

	inline xmap(const std::unordered_map<K*, V*>& other);
	inline xmap(std::unordered_map<K*, V*>&& other);

	inline xmap(const xmap<K*, V*>& other);
	inline xmap(xmap<K*, V*>&& other);

	inline void add_pair(const K* one, const V* two);
	// ======== INITALIZATION ========================================================================
	// ======== RETREVAL =============================================================================

	inline xvector<const K*> keys() const;
	inline xvector<V> values() const;
	inline K key_for_value(const V& input) const; // for Key-Value-Pairs
	inline xvector<const K*> cache() const;       // remember to allocate 

	inline V key(const K& input) const; // ------|
	inline V value_for(const K& input) const;//--|--all 3 are the same
	inline V at(const K& input) const; //--------|

	// ======== RETREVAL =============================================================================
	// ======== BOOLS ================================================================================

	inline bool has(const K& input) const;

	inline bool has_value(const V& input) const;          // for Key-Value-Pairs
	inline bool has_value_in_lists(const V& input) const; // for Key-list-Pairs

	inline bool operator()(const K& iKey) const;
	inline bool operator()(const K& iKey, const V& iValue) const;

	inline V operator[](const K& key) const;

	// ======== BOOLS ================================================================================
	// ======== Functional ===========================================================================
	inline xvector<const K*>* allocate_keys();
	inline xmap<const V*, const K*>* allocate_reverse_map();

	inline xvector<const K*> cached_keys() const; // remember to allocate 
	inline xmap<const V*, const K*> cached_rev_map() const; // remember to allocate 

	template<typename F>
	inline void sort(F func);

	template<typename F>
	inline void proc(F function);

	template<typename I, typename F>
	inline void proc(I& input, F function);

	inline void print() const;
	inline void print(int num) const;
	// ======== Functional ===========================================================================
};

// ======== INITALIZATION ========================================================================
template<typename K, typename V>
inline xmap<K*, V*>::xmap()
{
}

template<typename K, typename V>
inline xmap<K*, V*>::~xmap()
{
	if (m_keys != nullptr)
		delete m_keys;

	if (m_rev_map != nullptr)
		delete m_rev_map;
}

template<typename K, typename V>
inline xmap<K*, V*>::xmap(std::initializer_list<std::pair<K*, V*>> init)
{
	//for (const std::pair<K*, V*>& val : init) // for std::map
	//	this->insert(val);
	this->insert(init.begin(), init.end());

}

template<typename K, typename V>
inline xmap<K*, V*>::xmap(const std::unordered_map<K*, V*>& other)
	: std::unordered_map<K*, V*>(other.begin(), other.end())
{}

template<typename K, typename V>
inline xmap<K*, V*>::xmap(std::unordered_map<K*, V*>&& other)
	: std::unordered_map<K*, V*>(other.begin(), other.end())
{}

template<typename K, typename V>
inline xmap<K*, V*>::xmap(const xmap<K*, V*>& other)
	: std::unordered_map<K*, V*>(other.begin(), other.end())
{}

template<typename K, typename V>
inline xmap<K*, V*>::xmap(xmap<K*, V*>&& other)
	: std::unordered_map<K*, V*>(other.begin(), other.end())
{}

template<typename K, typename V>
inline void xmap<K*, V*>::add_pair(const K* one, const V* two)
{
	this->insert(std::make_pair(one, two));
}
// ======== INITALIZATION ========================================================================
// ======== RETREVAL =============================================================================

template<typename K, typename V>
inline xvector<const K*> xmap<K*, V*>::keys() const
{
	xvector<const K*> vec;
	for (typename std::unordered_map<K*, V*>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
		vec.push_back(iter->first);
	return vec;
}

template<typename K, typename V>
inline xvector<V> xmap<K*, V*>::values() const
{
	xvector<V> vec;
	for (typename std::unordered_map<K*, V*>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
		vec.push_back(iter->second);
	return vec;
}

template<typename K, typename V>
inline xvector<const K*> xmap<K*, V*>::cache() const
{
	if (m_keys == nullptr)
		return xvector<const K*>();
	return *m_keys;
}

template<typename K, typename V>
inline V xmap<K*, V*>::key(const K& input) const
{
	for (typename std::unordered_map<K*, V*>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
		if (*iter->first == input)
			return *iter->second;
	}
	return V();
}
template<typename K, typename V>
inline V xmap<K*, V*>::value_for(const K& input) const
{
	for (typename std::unordered_map<K*, V*>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
		if (*iter->first == input)
			return *iter->second;
	}
	return V();
}
template<typename K, typename V>
inline V xmap<K*, V*>::at(const K& input) const
{
	if (this->size() == 0)
		return V();
	for (typename std::unordered_map<K*, V*>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
		if (*iter->first == input)
			return *iter->second;
	}
	return V();
}
// ======== RETREVAL =============================================================================
// ======== BOOLS ================================================================================

template<typename K, typename V>
inline bool xmap<K*, V*>::has(const K& input) const
{
	for (typename std::unordered_map<K*, V*>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
		if (*iter->first == input)
			return true;
	}
	return false;
}

template<typename K, typename V>
inline bool xmap<K*, V*>::operator()(const K& iKey) const
{
	if (this->has(iKey))
		return true;
	else
		return false;
}

template<typename K, typename V>
inline bool xmap<K*, V*>::operator()(const K& iKey, const V& iValue) const
{
	if (this->key(iKey) == iValue)
		return true;
	else
		return false;
}


template<typename K, typename V>
inline V xmap<K*, V*>::operator[](const K& key) const {

	return this->at(key);
}

// ======== BOOLS ================================================================================
// ======== Functional ===========================================================================
template<typename K, typename V>
inline xvector<const K*>* xmap<K*, V*>::allocate_keys()
{
	if (m_keys == nullptr)
		m_keys = new xvector<const K*>;
	else
		m_keys->clear();

	for (typename std::unordered_map<K*, V*>::iterator iter = this->begin(); iter != this->end(); ++iter)
		m_keys->push_back(iter->first);
	return m_keys;
}


template<typename K, typename V>
inline xmap<const V*, const K*>* xmap<K*, V*>::allocate_reverse_map()
{
	if (m_rev_map == nullptr)
		m_rev_map = new xmap<const V*, const K*>;
	else
		m_keys->clear();

	for (typename std::unordered_map<K*, V*>::iterator iter = this->begin(); iter != this->end(); ++iter)
		m_rev_map->add_pair(iter->second, iter->first);

	return m_rev_map;
}


template<typename K, typename V>
inline xvector<const K*> xmap<K*, V*>::cached_keys() const
{
	if (m_keys == nullptr)
		return xvector<const K*>();
	return *m_keys;
}

template<typename K, typename V>
inline xmap<const V*, const K*> xmap<K*, V*>::cached_rev_map() const
{
	return *m_rev_map;
}



template<typename K, typename V>
template<typename F>
inline void xmap<K*, V*>::sort(F func)
{
	std::sort(m_keys->begin(), m_keys->end(), func);
}

template<typename K, typename V>
template<typename F>
inline void xmap<K*, V*>::proc(F function)
{
	for (typename std::unordered_map<K*, V*>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
		function(iter->first, iter->second);
}

template<typename K, typename V>
template<typename I, typename F>
inline void xmap<K*, V*>::proc(I& input, F function)
{
	for (typename std::unordered_map<K*, V*>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
		function(input, *iter);
}

template<typename K, typename V>
inline void xmap<K*, V*>::print() const
{
	typename std::unordered_map<K*, V*>::const_iterator iter;
	size_t max_size = 0;
	for (iter = this->begin(); iter != this->end(); ++iter) {
		if (iter->first->size() > max_size)
			max_size = iter->first->size();
	}

	for (iter = this->begin(); iter != this->end(); ++iter)
		std::cout << *iter->first << std::string(max_size - iter->first->size() + 3, '.') << *iter->second << std::endl;
}

template<typename K, typename V>
inline void xmap<K*, V*>::print(int num) const
{
	this->print();
	char* new_lines = static_cast<char*>(calloc(static_cast<size_t>(num) + 1, sizeof(char)));
	// calloc was used instead of "new" because "new" would give un-wanted after-effects.
	for (int i = 0; i < num; i++)
#pragma warning(suppress:6011) // we are derferencing a pointer, but assigning it a value at the same time
		new_lines[i] = '\n';
	std::cout << new_lines;
	free(new_lines);
}
// ======== Functional ===========================================================================
