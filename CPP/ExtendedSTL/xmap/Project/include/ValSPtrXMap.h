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

#include "BaseXMap.h"

template<typename K, typename V, typename H>
class xmap<K,xp<V>,H> : public BaseXMap<K,xp<V>,H>
{
public:
    // ======== INITALIZATION ========================================================================
    using BaseXMap<K,xp<V>,H>::BaseXMap;
    using BaseXMap<K,xp<V>,H>::operator=;

    //inline xmap();

    //inline xmap(const xmap<K,xp<V>,H>& other);
    //inline xmap(xmap<K,xp<V>,H>&& other) noexcept;

    //inline xmap(const std::unordered_map<K, xp<V>>& other);
    //inline xmap(std::unordered_map<K, xp<V>>&& other) noexcept;

    //inline xmap(const std::map<K, xp<V>>& other);
    //inline xmap(std::map<K, xp<V>>&& other) noexcept;

    template<typename ...R>
    inline void AddPairEmplace(const K& one, R&& ...ValueArgs);

    inline void AddPair(const K& one, const xp<V>& two);
    inline void AddPair(const K& one,       xp<V>&& two);

    // ======== INITALIZATION ========================================================================
    // ======== RETREVAL =============================================================================

    inline constexpr xvector<K> GetKeys() const;
    inline constexpr xvector<xp<V>> GetValues() const;

    inline const V& Key(const K& input) const; // ----------------|
    inline constexpr K& GetKeyFromValue(const V& input) const; //-| // for Key-Value-Pairs
    inline constexpr V& GetValueFrom(const K& input) const;//-----|--all 3 are the same

    inline const xp<V>& KeyPtr(const K& input) const;
    inline xp<V>& KeyPtr(const K& input);
    inline V& Key(const K& input); // ------------|
    inline K& GetKeyFromValue(const V& input);// -| for Key-Value-Pairs
    inline V& GetValueFrom(const K& input) ;//----|--all 3 are the same

    // ======== RETREVAL =============================================================================
    // ======== BOOLS ================================================================================

    inline bool Has(const K& input) const; // has implies key

    inline bool HasValue(const V& input) const;          // for Key-Value-Pairs
    inline bool HasValueInLists(const V& input) const; // for Key-list-Pairs

    inline bool operator()(const K& iKey) const;
    inline bool operator()(const K& iKey, const V& iValue) const;

    template<typename O>
    inline void operator=(const O& other);
    template<typename O>
    inline void operator=(O&& other);

    //inline const V& operator[](const K& key) const;
    //inline V& operator[](const K& key);

    inline void operator+=(const xmap<K,xp<V>,H>& other);
    inline constexpr xmap<K,xp<V>,H> operator+(const xmap<K,xp<V>,H>& other) const;

    inline size_t Size() const;

    // ======== BOOLS ================================================================================
    // ======== Functional ===========================================================================

    template<typename F>
    inline xvector<K> GetSortedKeys(F&& Function) const;
    inline xvector<K> GetSortedKeys() const;
    template<typename F>
    inline xvector<xp<V>>  GetSortedValues(F&& Function) const;
    inline xvector<xp<V>>  GetSortedValues() const;
    inline std::set<xp<V>> GetSortedValuesSet() const;
    template<typename F>
    inline xvector<K> GetReverseSortedKeys(F&& Function) const;
    inline xvector<K> GetReverseSortedKeys() const;
    inline xvector<xp<V>> GetReverseSortedValues() const;

    template<typename F, typename... A>
    inline void Proc(F&& function, A&& ...Args);
    template<typename F, typename... A>
    inline void ThreadProc(F&& function, A&& ...Args);

    // R = Return Type  ||  T = Nexus Type
    template<typename R = K, typename F, typename ...A>
    inline constexpr xvector<R> ForEach(F&& function, A&& ...Args) const;
    template<typename T = K, typename R = T, typename F, typename ...A>
    inline xvector<R> ForEachThread(F&& function, A&& ...Args) const;

    inline constexpr std::map<K, xp<V>> ToStdMap() const;
    inline constexpr std::unordered_map<K, xp<V>> ToStdUnorderedMap() const;

    inline void Remove(const K& Key);

    inline void Print() const;
    inline void Print(int num) const;
    // ======== Functional ===========================================================================
};

// ======== INITALIZATION ========================================================================

//template<typename K, typename V, typename H>
//inline xmap<K,xp<V>,H>::xmap()
//{
//}
//template<typename K, typename V, typename H>
//inline xmap<K,xp<V>,H>::xmap(const xmap<K,xp<V>,H>& other) :
//    std::unordered_map<K, xp<V>>(other.begin(), other.end()) {}
//template<typename K, typename V, typename H>
//inline xmap<K,xp<V>,H>::xmap(xmap<K,xp<V>,H>&& other) noexcept :
//    std::unordered_map<K, xp<V>>(std::move(other)) { }
//
//template<typename K, typename V, typename H>
//inline xmap<K,xp<V>,H>::xmap(const std::unordered_map<K, xp<V>>& other) :
//    std::unordered_map<K, xp<V>>(other.begin(), other.end()) {}
//template<typename K, typename V, typename H>
//inline xmap<K,xp<V>,H>::xmap(std::unordered_map<K, xp<V>>&& other) noexcept :
//    std::unordered_map<K, xp<V>>(std::move(other)) { }
//
//template<typename K, typename V, typename H>
//inline xmap<K,xp<V>,H>::xmap(const std::map<K, xp<V>>& other) :
//    std::unordered_map<K, xp<V>>(other.begin(), other.end()){}
//template<typename K, typename V, typename H>
//inline xmap<K,xp<V>,H>::xmap(std::map<K, xp<V>>&& other) noexcept :
//    std::unordered_map<K, xp<V>>(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end())) { }

namespace RA
{
    template<typename K1, typename K2, typename V>
    void XMapAddKeyKeyValue(xmap<K1, xmap<K2, V>>& FMap, const K1& FKey1, const K2& FKey2, const V& FValue)
    {
        if (FMap.Has(FKey1))
            FMap.Key(FKey1).AddPair(FKey2, FValue);
        else
            FMap.AddPair(FKey1, { {FKey2, FValue} });
    }

    template<typename K, typename V>
    void XMapAddKeyArrIdx(xmap<K, xvector<V>>& FMap, const K& FKey, const V& FValue)
    {
        if (FMap.Has(FKey))
            FMap.Key(FKey).push_back(FValue);
        else
            FMap.AddPair(FKey, { FValue });
    }

    template<typename K1, typename K2, typename V>
    void XMapAddKeyKeyArrIdx(xmap<K1, xmap<K2, xp<V>>>& FMap, const K1& FKey1, const K2& FKey2, const V& FValue)
    {
        if (FMap.Has(FKey1))
        {
            if (FMap.Key(FKey1).Has(FKey2))
                FMap.Key(FKey1).Key(FKey2).push_back(FValue);
            else
                FMap.Key(FKey1).AddPair(FKey2, { FValue });
        }
        else
            FMap.AddPair(FKey1, { { FKey2, { FValue }} });
    }
}

template<typename K, typename V, typename H>
template<typename ...R>
inline void xmap<K,xp<V>,H>::AddPairEmplace(const K& one, R&& ...ValueArgs)
{
    this->insert_or_assign(one, RA::MakeShared(std::forward(ValueArgs)...));
}

template<typename K, typename V, typename H>
inline void xmap<K,xp<V>,H>::AddPair(const K& one, const xp<V>&  two) {
    this->insert_or_assign(one, two);
}
template<typename K, typename V, typename H>
inline void xmap<K,xp<V>,H>::AddPair(const K& one,       xp<V>&& two) {
    this->insert_or_assign(one, std::move(two));
}

// ======== INITALIZATION ========================================================================
// ======== RETREVAL =============================================================================

template<typename K, typename V, typename H>
inline constexpr xvector<K> xmap<K,xp<V>,H>::GetKeys() const
{
    xvector<K> vec;
    for (typename std::unordered_map<K, xp<V>>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        vec.push_back(iter->first);
    return vec;
}

template<typename K, typename V, typename H>
inline constexpr xvector<xp<V>> xmap<K,xp<V>,H>::GetValues() const
{
    xvector<xp<V>> vec;
    for (typename std::unordered_map<K, xp<V>>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        vec.push_back(iter->second);
    return vec;
}

template<typename K, typename V, typename H>
inline const V& xmap<K,xp<V>,H>::Key(const K& input) const
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    return it->second.Get();
}

template<typename K, typename V, typename H>
inline constexpr K& xmap<K,xp<V>,H>::GetKeyFromValue(const V& input) const
{
    for (typename std::unordered_map<K, xp<V>>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (iter->second.Get() == input)
            return iter->first;
    }
    ThrowIt("Has No Value: ", input);
}

template<typename K, typename V, typename H>
inline constexpr V& xmap<K,xp<V>,H>::GetValueFrom(const K& input) const
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    else
        return it->second.Get();
}

template<typename K, typename V, typename H>
inline const xp<V>& xmap<K, xp<V>, H>::KeyPtr(const K& input) const
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    return it->second;
}

template<typename K, typename V, typename H>
inline xp<V>& xmap<K, xp<V>, H>::KeyPtr(const K& input)
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    return it->second;
}

template<typename K, typename V, typename H>
inline V& xmap<K,xp<V>,H>::Key(const K& input)
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    return it->second.Get();
}

template<typename K, typename V, typename H>
inline K& xmap<K,xp<V>,H>::GetKeyFromValue(const V& input)
{
    for (typename std::unordered_map<K, xp<V>>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (iter->second.Get() == input)
            return iter->first;
    }
    ThrowIt("Has No Value: ", input);
}

template<typename K, typename V, typename H>
inline V& xmap<K,xp<V>,H>::GetValueFrom(const K& input)
{
    auto it = this->find(input);
    if (it == this->end())
        ThrowIt("No Key: ", input);
    else
        return it->second.Get();
}

// ======== RETREVAL =============================================================================
// ======== BOOLS ================================================================================

template<typename K, typename V, typename H>
inline bool xmap<K,xp<V>,H>::Has(const K& input) const
{
    const auto it = this->find(input);
    if (it == this->end())
        return false;
    else
        return true;
}

template<typename K, typename V, typename H>
inline bool xmap<K,xp<V>,H>::HasValue(const V& input) const
{
    for (typename std::unordered_map<K, xp<V>>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        if (iter->second.Get() == input)
            return true;
    }
    return false;
}

template<typename K, typename V, typename H>
inline bool xmap<K,xp<V>,H>::HasValueInLists(const V& input) const
{
    for (typename std::unordered_map<K, xp<V>>::const_iterator map_iter = this->begin(); map_iter != this->end(); ++map_iter) {
        for (typename V::iterator val_iter = map_iter->second.Get()->begin(); val_iter != map_iter->second.Get()->end(); val_iter++) {
            if (input == *val_iter)
                return true;
        }
    }
    return false;
}


template<typename K, typename V, typename H>
inline bool xmap<K,xp<V>,H>::operator()(const K& iKey) const
{
    if (this->Has(iKey))
        return true;
    else
        return false;
}

template<typename K, typename V, typename H>
inline bool xmap<K,xp<V>,H>::operator()(const K& iKey, const V& iValue) const
{
    if (this->Key(iKey) == iValue)
        return true;
    else
        return false;
}

template<typename K, typename V, typename H>
template<typename O>
inline void xmap<K,xp<V>,H>::operator=(const O& other)
{
    this->clear();
    this->insert(other.begin(), other.end());
}

template<typename K, typename V, typename H>
template<typename O>
inline void xmap<K,xp<V>,H>::operator=(O&& other)
{
    this->clear();
    this->insert(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
    //this->insert(this->begin(), other.begin(), other.end());
}

//template<typename K, typename V, typename H>
//inline const V& xmap<K,xp<V>,H>::operator[](const K& key) const
//{
//    return (*this)[key];
//}
//
//template<typename K, typename V, typename H>
//inline V& xmap<K,xp<V>,H>::operator[](const K& key)
//{
//    return (*this)[key];
//}

template<typename K, typename V, typename H>
inline void xmap<K,xp<V>,H>::operator+=(const xmap<K,xp<V>,H>& other)
{
    this->insert(other.begin(), other.end());
}

template<typename K, typename V, typename H>
inline constexpr xmap<K,xp<V>,H> xmap<K,xp<V>,H>::operator+(const xmap<K,xp<V>,H>& other) const
{
    xmap<K,xp<V>,H> rmap = *this;
    rmap += other;
    return rmap;
}

template<typename K, typename V, typename H>
inline size_t xmap<K,xp<V>,H>::Size() const
{
    return this->size();
}

// ======== BOOLS ================================================================================
// ======== Functional ===========================================================================

template<typename K, typename V, typename H>
template<typename F>
inline xvector<K> xmap<K,xp<V>,H>::GetSortedKeys(F&& Function) const
{
    xvector<K> Sorted;
    for (typename xmap<K,xp<V>,H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).first;
    }
    return Sorted.Sort(Function);
}

template<typename K, typename V, typename H>
inline xvector<K> xmap<K,xp<V>,H>::GetSortedKeys() const
{
    xvector<K> Sorted;
    for (typename xmap<K,xp<V>,H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).first;
    }
    return Sorted.Sort();
}

template<typename K, typename V, typename H>
template<typename F>
inline xvector<xp<V>> xmap<K,xp<V>,H>::GetSortedValues(F&& Function) const
{
    xvector<xp<V>> Sorted;
    for (typename xmap<K,xp<V>,H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).second;
    }
    return Sorted.Sort(Function);
}

template<typename K, typename V, typename H>
inline xvector<xp<V>> xmap<K,xp<V>,H>::GetSortedValues() const
{
    xvector<xp<V>> Sorted;
    for (typename xmap<K,xp<V>,H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).second;
    }
    return Sorted.Sort();
}

template<typename K, typename V, typename H>
inline std::set<xp<V>> xmap<K,xp<V>,H>::GetSortedValuesSet() const
{
    std::set<xp<V>> Sorted;
    for (typename xmap<K,xp<V>,H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted.emplace((*iter).second);
    }
    return Sorted;
}
// -------------------------------------------------------------------------------------------------
template<typename K, typename V, typename H>
template<typename F>
inline xvector<K> xmap<K,xp<V>,H>::GetReverseSortedKeys(F&& Function) const
{
    xvector<K> Sorted;
    for (typename xmap<K,xp<V>,H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).first;
    }
    return Sorted.ReverseSort(Function);
}

template<typename K, typename V, typename H>
inline xvector<K> xmap<K,xp<V>,H>::GetReverseSortedKeys() const
{
    xvector<K> Sorted;
    for (typename xmap<K,xp<V>,H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).first;
    }
    return Sorted.ReverseSort();
}

template<typename K, typename V, typename H>
inline xvector<xp<V>> xmap<K,xp<V>,H>::GetReverseSortedValues() const
{
    xvector<xp<V>> Sorted;
    for (typename xmap<K,xp<V>,H>::const_iterator iter = this->begin(); iter != this->end(); ++iter) {
        Sorted << (*iter).second;
    }
    return Sorted.ReverseSort();
}

template<typename K, typename V, typename H>
template<typename F, typename... A>
inline void xmap<K,xp<V>,H>::Proc(F&& function, A&& ...Args)
{
    for (typename xmap<K,xp<V>,H>::iterator iter = this->begin(); iter != this->end(); ++iter) {
        if(function(iter->first, iter->second.Get(), Args...))
            break;
    }
}

template<typename K, typename V, typename H>
template<typename F, typename... A>
inline void xmap<K,xp<V>,H>::ThreadProc(F&& function, A&& ...Args)
{
    for (typename xmap<K,xp<V>,H>::iterator iter = this->begin(); iter != this->end(); ++iter)
        Nexus<>::AddJobPair(function, iter->first, iter->second.Get(), std::ref(Args)...);
}

template<typename K, typename V, typename H>
template<typename R, typename F, typename ...A>
inline constexpr xvector<R> xmap<K,xp<V>,H>::ForEach(F&& function, A&& ...Args) const
{
    xvector<R> vret;
    for (typename std::unordered_map<K, xp<V>>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        vret.push_back(function(iter->first, iter->second.Get(), Args...));
    return vret;
}

template<typename K, typename V, typename H>
template<typename T, typename R, typename F, typename ...A>
inline xvector<R> xmap<K,xp<V>,H>::ForEachThread(F&& function, A&& ...Args) const
{
    Nexus<T> td;
    for (typename std::unordered_map<K, xp<V>>::const_iterator iter = this->begin(); iter != this->end(); ++iter)
        td.AddJobPair(function, iter->first, iter->second.Get(), std::ref(Args)...);

    td.WaitAll();
    xvector<R> vret;
    vret.reserve(td.Size());

    for (size_t i = 0; i < td.Size(); i++)
        vret << td.GetWithoutProtection(i).GetValue();

    td.Clear();
    return vret;
}

template<typename K, typename V, typename H>
inline constexpr std::map<K, xp<V>> xmap<K,xp<V>,H>::ToStdMap() const
{
    std::map<K, xp<V>> stdmap;
    stdmap.insert(this->begin(), this->end());
    return stdmap;
}

template<typename K, typename V, typename H>
inline constexpr std::unordered_map<K, xp<V>> xmap<K,xp<V>,H>::ToStdUnorderedMap() const
{
    return *this;
}



template<typename K, typename V, typename H>
inline void xmap<K, xp<V>, H>::Remove(const K& Key)
{
    auto it = this->find(Key);
    if (it != this->end()) {
        this->erase(it);
    }
}

template<typename K, typename V, typename H>
inline void xmap<K,xp<V>,H>::Print() const
{
    typename std::unordered_map<K, xp<V>>::const_iterator iter;
    size_t max_size = 0;
    for (iter = this->begin(); iter != this->end(); ++iter) {
        if (iter->first.size() > max_size)
            max_size = iter->first.size();
    }

    for (iter = this->begin(); iter != this->end(); ++iter)
        std::cout << iter->first << std::string(max_size - iter->first.size() + 3, '.') << iter->second.Get() << std::endl;
}

template<typename K, typename V, typename H>
inline void xmap<K,xp<V>,H>::Print(int num) const
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
