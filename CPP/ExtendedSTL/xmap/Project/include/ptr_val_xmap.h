#pragma once

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


template<typename K, typename V> class xmap;

#include "xmap.h"

template<typename K, typename V>
class xmap<K*,V> : public std::unordered_map<K*,V>
{
private:
    // All of the pointers are allocated on an "As-Needed Basis"
    xvector<const K*>* m_keys = nullptr;             // used for setting orders to your keys 
    xmap<const V*, const K*>* m_rev_map = nullptr;   // go from KVPs to VKPs

public:
    using std::unordered_map<K*, V>::unordered_map;
    // ======== INITALIZATION ========================================================================
    inline xmap();
    inline ~xmap();

    inline xmap(const xmap<K*, V>& other);
    inline xmap(xmap<K*, V>&& other) noexcept;

    inline xmap(const std::unordered_map<K*, V>& other);
    inline xmap(std::unordered_map<K*, V>&& other) noexcept;

    inline xmap(const std::map<K*, V>& other);
    inline xmap(std::map<K*, V>&& other) noexcept;

    inline void AddPair(K* one, const V& two);
    // ======== INITALIZATION ========================================================================
    // ======== RETREVAL =============================================================================

    inline constexpr xvector<K*> GetKeys() const;
    inline constexpr xvector<V> GetValues() const;
    inline constexpr xvector<const K*> GetCache() const; // remember to allocate 

    inline constexpr V Key(const K& input) const; // ------|
    inline constexpr V GetValueFrom(const K& input) const;//--|--all 3 are the same
    inline constexpr V At(const K& input) const; //--------|

    // ======== RETREVAL =============================================================================
    // ======== BOOLS ================================================================================

    inline constexpr bool Has(const K& input) const;

    inline constexpr bool operator()(const K& iKey) const;
    inline constexpr bool operator()(const K& iKey, const V& iValue) const;

    template<typename O>
    inline void operator=(const O& other);
    template<typename O>
    inline void operator=(O&& other);

    inline const V& operator[](const K& key) const;
    inline V& operator[](const K& key);

    inline void operator+=(const xmap<K*, V>& other);
    inline xmap<K*, V> operator+(const xmap<K*, V>& other) const;

    inline size_t Size() const;

    // ======== BOOLS ================================================================================
    // ======== Functional ===========================================================================
    inline constexpr xvector<const K*>* AllocateKeys();
    inline xmap<const V*, const K*>* AllocateReverseMap();

    inline constexpr xvector<const K*> GetCachedKeys() const; // remember to allocate 
    inline xmap<const V*, const K*> GetCachedRevMap() const; // remember to allocate 

    template<typename F>
    inline void Sort(F func);

    template<typename F, typename... A>
    inline void Proc(F&& function, A&& ...Args);
    template<typename F, typename... A>
    inline void ThreadProc(F&& function, A&& ...Args);

    // R = Return Type  ||  T = Nexus Type
    template<typename R = K, typename F, typename ...A>
    inline constexpr xvector<R> ForEach(F&& function, A&& ...Args);
    template<typename T = K, typename R = T, typename F, typename ...A>
    inline xvector<R> ForEachThread(F&& function, A&& ...Args);

    inline constexpr std::map<K*, V> ToStdMap() const;
    inline constexpr std::unordered_map<K*, V> ToStdUnorderedMap() const;

    inline void Print() const;
    inline void Print(int num) const;
    // ======== Functional ===========================================================================
};

// ======== INITALIZATION ========================================================================

template<typename K, typename V>
inline xmap<K*, V>::xmap()
{
}

template<typename K, typename V>
inline xmap<K*, V>::~xmap()
{
    if (m_keys != nullptr)
        delete m_keys;

    if (m_rev_map != nullptr)
        delete m_rev_map;
}

template<typename K, typename V>
inline xmap<K*, V>::xmap(const xmap<K*, V>& other) :
    std::unordered_map<K*, V>(other.begin(), other.end()) {}
template<typename K, typename V>
inline xmap<K*, V>::xmap(xmap<K*, V>&& other) noexcept :
    std::unordered_map<K*, V>(std::move(other)) { }

template<typename K, typename V>
inline xmap<K*, V>::xmap(const std::unordered_map<K*, V>& other) :
    std::unordered_map<K*, V>(other.begin(), other.end()) {}
template<typename K, typename V>
inline xmap<K*, V>::xmap(std::unordered_map<K*, V>&& other) noexcept :
    std::unordered_map<K*, V>(std::move(other)) { }

template<typename K, typename V>
inline xmap<K*, V>::xmap(const std::map<K*, V>& other) :
    std::unordered_map<K*, V>(other.begin(), other.end()) {}
template<typename K, typename V>
inline xmap<K*, V>::xmap(std::map<K*, V>&& other) noexcept :
    std::unordered_map<K*, V>(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end())) { }


template<typename K, typename V>
inline void xmap<K*, V>::AddPair(K* one, const V& two)
{
    this->insert(std::make_pair(one, two));
}

// ======== INITALIZATION ========================================================================
// ======== RETREVAL =============================================================================

template<typename K, typename V>
inline constexpr xvector<K*> xmap<K*, V>::GetKeys() const
{
    xvector<K*> vec;
    for (typename std::unordered_map<K*, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        vec.push_back(iter->first);
    return vec;
}

template<typename K, typename V>
inline constexpr xvector<V> xmap<K*, V>::GetValues() const
{
    xvector<V> vec;
    for (typename std::unordered_map<K*, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        vec.push_back(iter->second);
    return vec;
}

template<typename K, typename V>
inline constexpr xvector<const K*> xmap<K*, V>::GetCache() const
{
    if (m_keys == nullptr)
        return xvector<const K*>();
    return *m_keys;
}

template<typename K, typename V>
inline constexpr V xmap<K*, V>::Key(const K& input) const
{
    for (typename std::unordered_map<K*, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (*iter->first == input)
            return iter->second;
    }
    ThrowIt("No Key: ", input);
}
template<typename K, typename V>
inline constexpr V xmap<K*, V>::GetValueFrom(const K& input) const
{
    for (typename std::unordered_map<K*, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (*iter->first == input)
            return iter->second;
    }
    ThrowIt("No Key: ", input);
}
template<typename K, typename V>
inline constexpr V xmap<K*, V>::At(const K& input) const
{
    if (this->size() == 0)
        throw std::out_of_range("Map Size is Zero!");

    for (typename std::unordered_map<K*, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (*iter->first == input)
            return iter->second;
    }
    throw std::out_of_range("Key [" + input + "] Not Found!");
}
// ======== RETREVAL =============================================================================
// ======== BOOLS ================================================================================

template<typename K, typename V>
inline constexpr bool xmap<K*, V>::Has(const K& input) const
{
    for (typename std::unordered_map<K*, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (*iter->first == input)
            return true;
    }
    return false;
}

template<typename K, typename V>
inline constexpr bool xmap<K*, V>::operator()(const K& iKey) const
{
    if (this->Has(iKey))
        return true;
    else
        return false;
}

template<typename K, typename V>
inline constexpr bool xmap<K*, V>::operator()(const K& iKey, const V& iValue) const
{
    if (this->Key(iKey) == iValue)
        return true;
    else
        return false;
}

template<typename K, typename V>
template<typename O>
inline void xmap<K*, V>::operator=(const O& other)
{
    this->clear();
    this->insert(other.begin(), other.end());
}
template<typename K, typename V>
template<typename O>
inline void xmap<K*, V>::operator=(O&& other)
{
    this->clear();
    this->insert(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
}

template<typename K, typename V>
inline const V& xmap<K*, V>::operator[](const K& key) const
{
    for (typename std::unordered_map<K*, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (*iter->first == key)
            return iter->second;
    }
    ThrowIt("No Key Found: ", key);
}

template<typename K, typename V>
inline V& xmap<K*, V>::operator[](const K& key) 
{
    for (typename std::unordered_map<K*, V>::iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (*iter->first == key)
            return iter->second;
    }
    ThrowIt("No Key Found: ", key);
}

template<typename K, typename V>
inline void xmap<K*, V>::operator+=(const xmap<K*, V>& other)
{
    this->insert(other.begin(), other.end());
}

template<typename K, typename V>
inline xmap<K*, V> xmap<K*, V>::operator+(const xmap<K*, V>& other) const
{
    xmap<K, V> rmap = *this;
    rmap += other;
    return rmap;

}

template<typename K, typename V>
inline size_t xmap<K*, V>::Size() const
{
    return this->size();
}

// ======== BOOLS ================================================================================
// ======== Functional ===========================================================================
template<typename K, typename V>
inline constexpr xvector<const K*>* xmap<K*, V>::AllocateKeys()
{
    if (m_keys == nullptr)
        m_keys = new xvector<const K*>;
    else
        m_keys->clear();

    for (typename std::unordered_map<K*, V>::iterator iter = this->begin(); iter != this->end(); ++iter)
        m_keys->push_back(iter->first);
    return m_keys;
}


template<typename K, typename V>
inline xmap<const V*, const K*>* xmap<K*, V>::AllocateReverseMap()
{
    if (m_rev_map == nullptr)
        m_rev_map = new xmap<const V*, const K*>;
    else
        m_rev_map->clear();

    for (typename std::unordered_map<K*, V>::iterator iter = this->begin(); iter != this->end(); ++iter)
        m_rev_map->AddPair(&iter->second, iter->first);

    return m_rev_map;
}


template<typename K, typename V>
inline constexpr xvector<const K*> xmap<K*, V>::GetCachedKeys() const
{
    if (m_keys == nullptr)
        return xvector<const K*>();
    return *m_keys;
}

template<typename K, typename V>
inline xmap<const V*, const K*> xmap<K*, V>::GetCachedRevMap() const
{
    return *m_rev_map;
}


template<typename K, typename V>
template<typename F>
inline void xmap<K*, V>::Sort(F func)
{
    std::sort(m_keys->begin(), m_keys->end(), func);
}

template<typename K, typename V>
template<typename F, typename... A>
inline void xmap<K*, V>::Proc(F&& function, A&& ...Args)
{
    for (typename xmap<K*, V>::iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (function(iter->first, iter->second, Args...))
            break;
    }
}

template<typename K, typename V>
template<typename F, typename... A>
inline void xmap<K*, V>::ThreadProc(F&& function, A&& ...Args)
{
    for (typename xmap<K*, V>::iterator iter = this->begin(); iter != this->end(); ++iter)
        Nexus<>::AddJobPair(function, *iter->first, iter->second, Args...);
}

template<typename K, typename V>
template<typename R, typename F, typename ...A>
inline constexpr xvector<R> xmap<K*, V>::ForEach(F&& function, A&& ...Args)
{
    xvector<R> vret;
    for (typename std::unordered_map<K*, V>::iterator iter = this->begin(); iter != this->end(); ++iter)
        vret.push_back(function(iter->first, iter->second, Args...));
    return vret;
}

template<typename K, typename V>
template<typename T, typename R, typename F, typename ...A>
inline xvector<R> xmap<K*, V>::ForEachThread(F&& function, A&& ...Args)
{
    Nexus<T> td;

    for (typename std::unordered_map<K*, V>::iterator iter = this->begin(); iter != this->end(); ++iter)
        td.AddJobPair(function, *iter->first, iter->second, Args...);

    td.WaitAll();
    xvector<R> vret;
    vret.reserve(td.size());

    for (size_t i = 0; i < td.size(); i++)
        vret << td.GetWithoutProtection(i).GetValue();

    td.clear();
    return vret;
}

template<typename K, typename V>
inline constexpr std::map<K*, V> xmap<K*, V>::ToStdMap() const
{
    std::map<K*, V> stdmap;
    stdmap.insert(this->begin(), this->end());
    return stdmap;
}

template<typename K, typename V>
inline constexpr std::unordered_map<K*, V> xmap<K*, V>::ToStdUnorderedMap() const
{
    return *this;
}

template<typename K, typename V>
inline void xmap<K*, V>::Print() const
{
    typename std::unordered_map<K*, V>::const_iterator iter;
    size_t max_size = 0;
    for (iter = this->begin(); iter != this->end(); ++iter) {
        if (iter->first->size() > max_size)
            max_size = iter->first->size();
    }

    for (iter = this->begin(); iter != this->end(); ++iter)
        std::cout << *iter->first << std::string(max_size - iter->first->size() + 3, '.') << iter->second << std::endl;
}

template<typename K, typename V>
inline void xmap<K*, V>::Print(int num) const
{
    this->Print();
    char* new_lines = static_cast<char*>(calloc(static_cast<size_t>(num) + 1, sizeof(char)));
    // calloc was used instead of "new" because "new" would give un-wanted after-effects.
    for (int i = 0; i < num; i++)
#pragma warning(suppress:6011) // we are derferencing a pointer, but assigning it a value at the same time
        new_lines[i] = '\n';
    std::cout << new_lines;
    free(new_lines);
}
// ======== Functional ===========================================================================
