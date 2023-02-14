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

#include <sstream>
#include "BaseXMap.h"

template<typename K, typename V, typename H>
class xmap : public BaseXMap<K,V,H>
{
public:
    // ======== INITALIZATION ========================================================================
    using BaseXMap<K, V, H>::BaseXMap;
    // using BaseXMap<K, V, H>::operator=;

    //inline xmap();

    //inline xmap(const xmap<K, V, H>& other);
    //inline xmap(xmap<K, V, H>&& other) noexcept;

    //inline xmap(const std::unordered_map<K, V>& other);
    //inline xmap(std::unordered_map<K, V>&& other) noexcept;

    //inline xmap(const std::map<K, V>& other);
    //inline xmap(std::map<K, V>&& other) noexcept;

    inline void AddPair(const K& one, const V&  two) ValueMustCopy;
    inline void AddPair(const K& one,       V&& two) ValueMustCopy;

    // ======== INITALIZATION ========================================================================
    // ======== RETREVAL =============================================================================

    inline constexpr xvector<K> GetKeys() const;
    inline constexpr xvector<V> GetValues() const;

    inline constexpr K& GetKeyFromValue(const V& input) const; // for Key-Value-Pairs
    inline const V& Key(const K& input) const; // ------| can't be constexpr unless primative data type
    inline const V& KeyAnyCase(const K& input) const; // ------| can't be constexpr unless primative data type
    inline constexpr V& GetValueFrom(const K& input) const;//--|--all 3 are the same

    inline K& GetKeyFromValue(const V& input); // for Key-Value-Pairs
    inline V& Key(const K& input); // ------|
    inline V& KeyAnyCase(const K& input); // ------|
    inline V& GetValueFrom(const K& input);//--|--all 3 are the same

    // ======== RETREVAL =============================================================================
    // ======== BOOLS ================================================================================

    inline constexpr bool Has(const K& input) const; // has implies key
    inline constexpr bool Lacks(const K& input) const; // has implies key

    inline constexpr bool HasValue(const V& input) const;        // for Key-Value-Pairs
    inline constexpr bool HasValueInLists(const V& input) const; // for Key-list-Pairs

    inline constexpr bool operator()(const K& iKey) const;
    inline constexpr bool operator()(const K& iKey, const V& iValue) const;

    inline void operator=(const xmap<K, V, H>& other);

    template<typename OK, typename OV, typename OH>
    inline void operator=(xmap<OK, OV, OH>&& other) noexcept;

    //inline const V& operator[](const K& key) const;
    //inline V& operator[](const K& key);

    inline void operator+=(const xmap<K, V, H>& other);
    inline constexpr xmap<K, V, H> operator+(const xmap<K, V, H>& other) const;

    inline size_t Size() const;

    // ======== BOOLS ================================================================================
    // ======== Functional ===========================================================================

    template<typename F>
    inline xvector<K> GetSortedKeys(F&& Function) const;
    inline xvector<K> GetSortedKeys() const;
    template<typename F>
    inline xvector<V> GetSortedValues(F&& Function) const;
    inline xvector<V> GetSortedValues() const;
    template<typename F>
    inline xvector<K> GetReverseSortedKeys(F&& Function) const;
    inline xvector<K> GetReverseSortedKeys() const;
    inline xvector<V> GetReverseSortedValues() const;

    template<typename F, typename... A>
    inline void Proc(F&& function, A&& ...Args);
    template<typename F, typename... A>
    inline void ThreadProc(F&& function, A&& ...Args);

    // R = Return Type  ||  T = Nexus Type
    template<typename R = K, typename F, typename ...A>
    inline constexpr xvector<R> ForEach(F&& function, A&& ...Args) const;
    template<typename T = K, typename R = T, typename F, typename ...A>
    inline xvector<R> ForEachThread(F&& function, A&& ...Args) const;

    inline constexpr std::map<K, V> ToStdMap() const;
    inline constexpr std::unordered_map<K, V> ToStdUnorderedMap() const;

    inline void Remove(const K& Key);

    inline void Print() const;
    inline void Print(int num) const;
    // ======== Functional ===========================================================================
};

// ======== INITALIZATION ========================================================================

//template<typename K, typename V, typename H>
//inline xmap<K, V, H>::xmap()
//{
//}
//template<typename K, typename V, typename H>
//inline xmap<K, V, H>::xmap(const xmap<K, V, H>& other) :
//    std::unordered_map<K, V>(other.begin(), other.end()) {}
//template<typename K, typename V, typename H>
//inline xmap<K, V, H>::xmap(xmap<K, V, H>&& other) noexcept :
//    std::unordered_map<K, V>(std::move(other)) { }
//
//template<typename K, typename V, typename H>
//inline xmap<K, V, H>::xmap(const std::unordered_map<K, V>& other) :
//    std::unordered_map<K, V>(other.begin(), other.end()) {}
//template<typename K, typename V, typename H>
//inline xmap<K, V, H>::xmap(std::unordered_map<K, V>&& other) noexcept :
//    std::unordered_map<K, V>(std::move(other)) { }
//
//template<typename K, typename V, typename H>
//inline xmap<K, V, H>::xmap(const std::map<K, V>& other) :
//    std::unordered_map<K, V>(other.begin(), other.end()){}
//template<typename K, typename V, typename H>
//inline xmap<K, V, H>::xmap(std::map<K, V>&& other) noexcept :
//    std::unordered_map<K, V>(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end())) { }

template<typename K, typename V, typename H>
inline void xmap<K, V, H>::AddPair(const K& one, const V&  two) ValueMustCopy
{
    this->insert_or_assign(one, two);
}
template<typename K, typename V, typename H>
inline void xmap<K, V, H>::AddPair(const K& one,       V&& two) ValueMustCopy
{
    this->insert_or_assign(one, std::move(two));
}

// ======== INITALIZATION ========================================================================
// ======== RETREVAL =============================================================================

template<typename K, typename V, typename H>
inline constexpr xvector<K> xmap<K, V, H>::GetKeys() const
{
    xvector<K> vec;
    for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        vec.push_back(iter->first);
    return vec;
}

template<typename K, typename V, typename H>
inline constexpr xvector<V> xmap<K, V, H>::GetValues() const
{
    xvector<V> vec;
    for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        vec.push_back(iter->second);
    return vec;
}

template<typename K, typename V, typename H>
inline constexpr K& xmap<K, V, H>::GetKeyFromValue(const V& input) const
{
    for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (iter->second == input)
            return iter->first;
    }
    return K();
}

template<typename K, typename V, typename H>
inline const V& xmap<K, V, H>::Key(const K& input) const
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    return it->second;
}

template<typename K, typename V, typename H>
inline const V& xmap<K, V, H>::KeyAnyCase(const K& input) const
{
    for (auto& Pair : The)
    {
        if (Pair.first.ToLower() == input.ToLower())
            return Pair.second;
    }
    ThrowIt("No Key: ", input);
}

template<typename K, typename V, typename H>
inline constexpr V& xmap<K, V, H>::GetValueFrom(const K& input) const
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    else
        return it->second;
}


template<typename K, typename V, typename H>
inline K& xmap<K, V, H>::GetKeyFromValue(const V& input)
{
    for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (iter->second == input)
            return iter->first;
    }
    return K();
}

template<typename K, typename V, typename H>
inline V& xmap<K, V, H>::Key(const K& input)
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    return it->second;
}

template<typename K, typename V, typename H>
inline V& xmap<K, V, H>::KeyAnyCase(const K& input)
{
    for (auto& Pair : The)
    {
        if (Pair.first.ToLower() == input.ToLower())
            return Pair.second;
    }
    ThrowIt("No Key: ", input);
}

template<typename K, typename V, typename H>
inline V& xmap<K, V, H>::GetValueFrom(const K& input)
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    else
        return it->second;
}
// ======== RETREVAL =============================================================================
// ======== BOOLS ================================================================================

template<typename K, typename V, typename H>
inline constexpr bool xmap<K, V, H>::Has(const K& input) const
{
    auto it = this->find(input);
    if (it == this->end())
        return false;
    else
        return true;
}

template<typename K, typename V, typename H>
inline constexpr bool xmap<K, V, H>::Lacks(const K& input) const
{
    auto it = this->find(input);
    if (it == this->end())
        return true;
    else
        return false;
}

template<typename K, typename V, typename H>
inline constexpr bool xmap<K, V, H>::HasValue(const V& input) const
{
    for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (iter->second == input)
            return true;
    }
    return false;
}

template<typename K, typename V, typename H>
inline constexpr bool xmap<K, V, H>::HasValueInLists(const V& input) const
{
    for (typename std::unordered_map<K, V>::const_iterator map_iter = this->begin(); map_iter != this->end(); ++map_iter) {
        for (typename V::iterator val_iter = map_iter->second->begin(); val_iter != map_iter->second->end(); val_iter++) {
            if (input == *val_iter)
                return true;
        }
    }
    return false;
}


template<typename K, typename V, typename H>
inline constexpr bool xmap<K, V, H>::operator()(const K& iKey) const
{
    if (this->Has(iKey))
        return true;
    else
        return false;
}

template<typename K, typename V, typename H>
inline constexpr bool xmap<K, V, H>::operator()(const K& iKey, const V& iValue) const
{
    if (this->Key(iKey) == iValue)
        return true;
    else
        return false;
}

template<typename K, typename V, typename H>
inline void xmap<K, V, H>::operator=(const xmap<K, V, H>& other)
{
    this->clear();
    this->insert(other.begin(), other.end());
}


template<typename K, typename V, typename H>
template<typename OK, typename OV, typename OH>
inline void xmap<K, V, H>::operator=(xmap<OK, OV, OH>&& other) noexcept
{
    this->clear();
    this->insert(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
    //this->insert(this->begin(), other.begin(), other.end());
}

//template<typename K, typename V, typename H>
//inline void xmap<K, V, H>::operator=(xmap<K, V, H>&& other) noexcept
//{
//    this->clear();
//    this->insert(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
//    //this->insert(this->begin(), other.begin(), other.end());
//}

//template<typename K, typename V, typename H>
//inline const V& xmap<K, V, H>::operator[](const K& key) const
//{
//    return (*this)[key];
//}
//
//template<typename K, typename V, typename H>
//inline V& xmap<K, V, H>::operator[](const K& key)
//{
//    return (*this)[key];
//}

template<typename K, typename V, typename H>
inline void xmap<K, V, H>::operator+=(const xmap<K, V, H>& other)
{
    this->insert(other.begin(), other.end());
}

template<typename K, typename V, typename H>
inline constexpr xmap<K, V, H> xmap<K, V, H>::operator+(const xmap<K, V, H>& other) const
{
    xmap<K, V, H> rmap = *this;
    rmap += other;
    return rmap;
}

template<typename K, typename V, typename H>
inline size_t xmap<K, V, H>::Size() const
{
    return this->size();
}

// ======== BOOLS ================================================================================
// ======== Functional ===========================================================================

template<typename K, typename V, typename H>
template<typename F>
inline xvector<K> xmap<K, V, H>::GetSortedKeys(F&& Function) const
{
    xvector<K> Sorted;
    for (typename xmap<K, V, H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).first;
    }
    return Sorted.Sort(Function);
}

template<typename K, typename V, typename H>
inline xvector<K> xmap<K, V, H>::GetSortedKeys() const
{
    xvector<K> Sorted;
    for (typename xmap<K, V, H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).first;
    }
    return Sorted.Sort();
}

template<typename K, typename V, typename H>
template<typename F>
inline xvector<V> xmap<K, V, H>::GetSortedValues(F&& Function) const
{
    xvector<V> Sorted;
    for (typename xmap<K, V, H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).second;
    }
    return Sorted.Sort(Function);
}

template<typename K, typename V, typename H>
inline xvector<V> xmap<K, V, H>::GetSortedValues() const
{
    xvector<V> Sorted;
    for (typename xmap<K, V, H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).second;
    }
    return Sorted.Sort();
}
// -------------------------------------------------------------------------------------------------
template<typename K, typename V, typename H>
template<typename F>
inline xvector<K> xmap<K, V, H>::GetReverseSortedKeys(F&& Function) const
{
    xvector<K> Sorted;
    for (typename xmap<K, V, H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).first;
    }
    return Sorted.ReverseSort(Function);
}

template<typename K, typename V, typename H>
inline xvector<K> xmap<K, V, H>::GetReverseSortedKeys() const
{
    xvector<K> Sorted;
    for (typename xmap<K, V, H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).first;
    }
    return Sorted.ReverseSort();
}

template<typename K, typename V, typename H>
inline xvector<V> xmap<K, V, H>::GetReverseSortedValues() const
{
    xvector<V> Sorted;
    for (typename xmap<K, V, H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).second;
    }
    return Sorted.ReverseSort();
}

template<typename K, typename V, typename H>
template<typename F, typename... A>
inline void xmap<K, V, H>::Proc(F&& function, A&& ...Args)
{
    for (typename xmap<K, V, H>::iterator iter = this->begin(); iter != this->end(); ++iter) {
        if(function(iter->first, iter->second, Args...))
            break;
    }
}

template<typename K, typename V, typename H>
template<typename F, typename... A>
inline void xmap<K, V, H>::ThreadProc(F&& function, A&& ...Args)
{
    for (typename xmap<K, V, H>::iterator iter = this->begin(); iter != this->end(); ++iter)
        Nexus<>::AddJobPair(function, iter->first, iter->second, std::ref(Args)...);
}

template<typename K, typename V, typename H>
template<typename R, typename F, typename ...A>
inline constexpr xvector<R> xmap<K, V, H>::ForEach(F&& function, A&& ...Args) const
{
    xvector<R> vret;
    for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        vret.push_back(function(iter->first, iter->second, Args...));
    return vret;
}

template<typename K, typename V, typename H>
template<typename T, typename R, typename F, typename ...A>
inline xvector<R> xmap<K, V, H>::ForEachThread(F&& function, A&& ...Args) const
{
    Nexus<T> td;
    for (typename std::unordered_map<K, V>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        td.AddJobPair(function, iter->first, iter->second, std::ref(Args)...);

    td.WaitAll();
    xvector<R> vret;
    vret.reserve(td.Size());

    for (size_t i = 0; i < td.Size(); i++)
        vret << td.GetWithoutProtection(i).GetValue();

    td.Clear();
    return vret;
}

template<typename K, typename V, typename H>
inline constexpr std::map<K, V> xmap<K, V, H>::ToStdMap() const
{
    std::map<K, V> stdmap;
    stdmap.insert(this->begin(), this->end());
    return stdmap;
}

template<typename K, typename V, typename H>
inline constexpr std::unordered_map<K, V> xmap<K, V, H>::ToStdUnorderedMap() const
{
    return *this;
}

template<typename K, typename V, typename H>
inline void xmap<K, V, H>::Remove(const K& Key)
{
    auto it = this->find(Key);
    if (it != this->end()) {
        this->erase(it);
    }
}

template<typename K, typename V, typename H>
inline void xmap<K, V, H>::Print() const
{
    typename std::unordered_map<K, V>::const_iterator iter;
    size_t max_size = 0;
    std::ostringstream OSS;
    for (iter = this->begin(); iter != this->end(); ++iter) 
    {
        OSS << iter->first;
        if (OSS.str().size() > max_size)
            max_size = OSS.str().size();
        OSS.str("");
        OSS.clear();
    }

    for (iter = this->begin(); iter != this->end(); ++iter)
    {
        OSS << iter->first;
        std::cout << iter->first << std::string(max_size - OSS.str().size() + 3, '.') << iter->second << std::endl;
        OSS.str("");
        OSS.clear();
    }
}

template<typename K, typename V, typename H>
inline void xmap<K, V, H>::Print(int num) const
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
