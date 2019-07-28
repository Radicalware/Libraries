#pragma once

#include<map>
#include<unordered_map>
#include<initializer_list>
#include<utility>
#include<memory>

#include "xvector.h"
#include "xstring.h"

// todo: delete next 3 lines
#include <iostream>
using std::cout;
using std::endl;

template<typename K, typename V>
class xmap : public std::unordered_map<K,V>
{
private:
	xvector<const K*>* m_keys = nullptr; // TODO: NOT WORKING, VALUE DOES NOT SAVE/HOLD PERSISTENCE
	// Important!!! you must manually update m_keys
	// it won't update automatically to enhance speed. 

public:
	// ======== INITALIZATION ========================================================================
	inline xmap();
	inline ~xmap();

	inline xmap(std::initializer_list<std::pair<K, V>> init);

	inline xmap(const std::unordered_map<K, V>& other);
	inline xmap(std::unordered_map<K, V>&& other);

	inline xmap(const xmap<K, V>& other);
	inline xmap(xmap<K, V>&& other);


	// ======== INITALIZATION ========================================================================
	// ======== RETREVAL =============================================================================

	inline xvector<K> keys() const;
	inline xvector<V> values() const;
	inline xvector<const K*> keyStore() const; // remember to relocate
	inline V key(const K& input) const; // ------|
	inline V value_for(const K& input) const;//--|
	inline V at(const K& input); //--------------|

	// ======== RETREVAL =============================================================================
	// ======== BOOLS ================================================================================

	inline bool has(const K& input) const;
	//inline bool has(K&& input) const;
	//inline bool has(char const* input) const;
	inline bool operator()(const K& iKey) const;

	inline bool operator()(const K& iKey, const V& iValue) const;

	inline V operator[](const K& key);

	// ======== BOOLS ================================================================================
	// ======== Functional ===========================================================================
	inline xmap<K, V>* relocate(); // You must reallocate your pointers before using sort

	template<typename F>
	inline void sort(F func);

	template<typename F>
	inline void proc(F function);

	template<typename I, typename F>
	inline void proc(I& input, F function);
	// ======== Functional ===========================================================================
};

// ======== INITALIZATION ========================================================================
template<typename K, typename V>
inline xmap<K, V>::xmap()
{
}

template<typename K, typename V>
inline xmap<K, V>::~xmap()
{
	if (m_keys != nullptr)
		delete m_keys;
}

template<typename K, typename V>
inline xmap<K, V>::xmap(std::initializer_list<std::pair<K, V>> init)
{
	//for (const std::pair<K, V>& val : init) // for std::map
	//	this->insert(val);
	this->insert(init.begin(), init.end());

}

template<typename K, typename V>
inline xmap<K, V>::xmap(const std::unordered_map<K, V>& other)
	: std::unordered_map<K, V>(other.begin(), other.end())
{}

template<typename K, typename V>
inline xmap<K, V>::xmap(std::unordered_map<K, V>&& other)
	: std::unordered_map<K, V>(other.begin(), other.end())
{}

template<typename K, typename V>
inline xmap<K, V>::xmap(const xmap<K, V> & other)
	: std::unordered_map<K, V>(other.begin(), other.end())
{}

template<typename K, typename V>
inline xmap<K, V>::xmap(xmap<K, V> && other)
	: std::unordered_map<K, V>(other.begin(), other.end())
{}

// ======== INITALIZATION ========================================================================
// ======== RETREVAL =============================================================================

template<typename K, typename V>
inline xvector<K> xmap<K, V>::keys() const
{
	xvector<K> vec;
	for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
		vec.push_back(iter->first);
	return vec;
}

template<typename K, typename V>
inline xvector<V> xmap<K, V>::values() const
{
	xvector<V> vec;
	for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
		vec.push_back(iter->second);
	return vec;
}

template<typename K, typename V>
inline xvector<const K*> xmap<K, V>::keyStore() const
{
	if (m_keys == nullptr)
		return xvector<const K*>();
	return *m_keys;
}

template<typename K, typename V>
inline V xmap<K, V>::key(const K& input) const
{
	for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
		if (iter->first == input)
			return iter->second;
	}
	return V();
}
template<typename K, typename V>
inline V xmap<K, V>::value_for(const K& input) const
{
	for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
		if (iter->first == input)
			return iter->second;
	}
	return V();
}
template<typename K, typename V>
inline V xmap<K, V>::at(const K& input)
{
	if (this->size() == 0)
		return V();
	for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
		if (iter->first == input)
			return iter->second;
	}
	return V();
}
// ======== RETREVAL =============================================================================
// ======== BOOLS ================================================================================

template<typename K, typename V>
inline bool xmap<K, V>::has(const K& input) const
{
	for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
		if (iter->first == input)
			return true;
	}
	return false;
}

template<typename K, typename V>
inline bool xmap<K, V>::operator()(const K& iKey) const
{
	if (this->has(iKey))
		return true;
	else
		return false;
}

template<typename K, typename V>
inline bool xmap<K, V>::operator()(const K& iKey, const V& iValue) const
{
	if (this->key(iKey) == iValue)
		return true;
	else
		return false;
}


template<typename K, typename V>
inline V xmap<K, V>::operator[](const K& key) {

	return this->at(key);
}

// ======== BOOLS ================================================================================
// ======== Functional ===========================================================================
template<typename K, typename V>
inline xmap<K, V>* xmap<K, V>::relocate()
{
	if (m_keys == nullptr)
		m_keys = new xvector<const K*>;
	else
		m_keys->clear();

	for (typename std::unordered_map<K, V>::iterator iter = this->begin(); iter != this->end(); ++iter)
		m_keys->push_back(&iter->first);
	return this;
}

template<typename K, typename V>
template<typename F>
inline void xmap<K, V>::sort(F func)
{
	std::sort(m_keys->begin(), m_keys->end(), func);
}

template<typename K, typename V>
template<typename F>
inline void xmap<K, V>::proc(F function)
{
	for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
		function(iter->first, iter->second);
}

template<typename K, typename V>
template<typename I, typename F>
inline void xmap<K, V>::proc(I& input, F function)
{
	for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
		function(input, *iter);
}
// ======== Functional ===========================================================================
