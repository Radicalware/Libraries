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


#include "BaseXVector.h"
#include <vector>
#include <type_traits>

template<typename T>
class ValXVector : public BaseXVector<T>
{    
public:
    using BaseXVector<T>::BaseXVector;
    using BaseXVector<T>::operator=;
    using E = std::remove_const<T>::type; // E for Erratic

    inline T& At(const size_t Idx);
    inline const T& At(const size_t Idx) const;

    template<typename P = T> 
    inline bool Has(const P* item) const;
    inline bool Has(const T& item) const;
    inline bool Has(char const* item) const;
    template<typename F, typename ...A>
    inline bool HasTruth(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    inline const T& GetTruth(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    inline T& GetTruth(F&& Function, A&& ...Args);
    template<typename F, typename ...A>
    inline xvector<T> GetTruths(F&& Function, A&& ...Args) const;

    inline bool Lacks(const T& item) const;
    inline bool Lacks(char const* item) const;
    template<typename F, typename ...A>
    inline bool LacksTruth(F&& Function, A&& ...Args) const;

    template<typename L = xvector<T>>
    inline xvector<T> GetCommonItems(L& item);
    template<typename L = xvector<T>>
    inline xvector<T> GetCommonItems(L&& item);

    inline void operator<<(const T&  item);
    inline void operator<<(      T&& item);

    inline void AddCharStrings(int strC, char** strV);

    T& First(size_t Idx = 0);
    const T& First(size_t Idx = 0) const;

    T& Last(size_t Idx = 0);
    const T& Last(size_t Idx = 0) const;

    inline std::pair<T, T> GetPair() const;

    template<typename C>
    inline xvector<C> Convert() const;

    template<typename C, typename F>
    inline xvector<C> Convert(F function) const;

    template<typename N = unsigned int>
    inline xvector<xvector<T>> Split(N count = 1) const;

    inline void operator+=(const xvector<T>& other);
    inline xvector<T> operator+(const xvector<T>& other) const;

    size_t Size() const;

    inline void Organize();
    inline void RemoveDups();

    template<typename F>
    inline xvector<T>& Sort(F func);
    inline xvector<T>& Sort();
    inline xvector<T>& ReverseSort();
    inline xvector<T>& ReverseIt();

    inline xvector<T*> GetPtrs();

    template<typename F, typename... A>
    inline void Proc(F&& function, A&& ...Args);
    template<typename F, typename... A>
    inline void ProcThread(F&& function, A&& ...Args);

    // single threaded non-const
    template<typename N = E, typename F, typename ...A>
    inline xvector<N> ForEach(F&& function, A&& ...Args);
    template<typename K, typename V, typename F, typename ...A>
    inline std::unordered_map<K, V> ForEach(F&& function, A&& ...Args);
    // single threaded const
    template<typename N = E, typename F, typename ...A>
    inline xvector<N> ForEach(F&& function, A&& ...Args) const;
    template<typename K, typename V, typename F, typename ...A>
    inline std::unordered_map<K, V> ForEach(F&& function, A&& ...Args) const;
    // multi-threaded const & non-const
    template<typename N = E, typename F, typename... A>
    inline xvector<N> ForEachThread(F&& function, A&& ...Args);
    template<typename N = E, typename F, typename... A>
    inline xvector<N> ForEachThread(F&& function, A&& ...Args) const;
    template<typename K, typename V, typename F, typename ...A>
    inline std::unordered_map<K, V> ForEachThread(F&& function, A&& ...Args) const;

    template<typename N = E, typename F, typename... A>
    inline void StartTasks(F&& function, A&& ...Args);

    template<typename N = E>
    inline typename std::enable_if<!std::is_same<N, void>::value, xvector<N>>::type GetTasks() const;

    inline bool TasksCompleted() const;

    // =================================== DESIGNED FOR NUMERIC BASED VECTORS ===================================

    inline T GetSum(size_t FnSkipIdx = 0) const;
    inline T GetMul(size_t FnSkipIdx = 0) const;
    inline T GetAvg(size_t FnSkipIdx = 0) const;

    // =================================== DESIGNED FOR STRING  BASED VECTORS ===================================

    inline T Join(const T& str = "") const;
    inline T Join(const char str) const;
    inline T Join(const char* str) const;

    inline bool FullMatchOne(const re2::RE2& in_pattern) const;
    inline bool FullMatchOne(const std::string& in_pattern) const;
    inline bool FullMatchOne(std::string&& in_pattern) const;
    inline bool FullMatchOne(char const* in_pattern) const;

    inline bool FullMatchAll(const re2::RE2& in_pattern) const;
    inline bool FullMatchAll(const std::string& in_pattern) const;
    inline bool FullMatchAll(std::string&& in_pattern) const;
    inline bool FullMatchAll(char const* in_pattern) const;

    inline bool MatchOne(const re2::RE2& in_pattern) const;
    inline bool MatchOne(const std::string& in_pattern) const;
    inline bool MatchOne(std::string&& in_pattern) const;
    inline bool MatchOne(char const* in_pattern) const;

    inline bool MatchAll(const re2::RE2& in_pattern) const;
    inline bool MatchAll(const std::string& in_pattern) const;
    inline bool MatchAll(std::string&& in_pattern) const;
    inline bool MatchAll(char const* in_pattern) const;

    inline xvector<T> Take(const re2::RE2& in_pattern) const; // -- // TODO rename to FindOne && FindAll (lambda)
    inline xvector<T> Take(const std::string& in_pattern) const;
    inline xvector<T> Take(std::string&& in_pattern) const;
    inline xvector<T> Take(char const* in_pattern) const;

    inline xvector<T> Remove(const re2::RE2& in_pattern) const;
    inline xvector<T> Remove(const std::string& in_pattern) const;
    inline xvector<T> Remove(std::string&& in_pattern) const;
    inline xvector<T> Remove(char const* in_pattern) const;

    inline xvector<T> SubAll(const re2::RE2& in_pattern, const std::string& replacement) const;
    inline xvector<T> SubAll(const std::string& in_pattern, const std::string& replacement) const;
    inline xvector<T> SubAll(std::string&& in_pattern, std::string&& replacement) const;
    inline xvector<T> SubAll(char const* in_pattern, char const* replacement) const;
    
    // double was chose to hold long signed and unsigned values
    inline xvector<T> operator()(long double x = 0, long double y = 0, long double z = 0, const char removal_method = 's') const;
    // s = slice perserves values you land on 
    // d = dice  removes values you land on
    // s/d only makes a difference if you modify the 'z' value

    // =================================== DESIGNED FOR STRING BASED VECTORS ===================================
};
// =============================================================================================================

template<typename T>
inline T& ValXVector<T>::At(const size_t Idx)
{
    if (Idx >= Size())
        throw "Index Out Of Range";
    return (*this)[Idx];
}

template<typename T>
inline const T& ValXVector<T>::At(const size_t Idx) const
{
    if (Idx >= Size())
        throw "Index Out Of Range";
    return (*this)[Idx];
}

template<typename T>
template<typename P>
inline bool ValXVector<T>::Has(const P* item) const
{
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (**it == *item)
            return true;
    }
    return false;
}

template<typename T>
inline bool ValXVector<T>::Has(const T& item) const{
    return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
inline bool ValXVector<T>::Has(char const* item) const {
    return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
template<typename F, typename ...A>
inline bool ValXVector<T>::HasTruth(F&& Function, A&& ...Args) const
{
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (Function(*it, std::forward<A>(Args)...))
            return true;
    }
    return false;
}

template<typename T>
template<typename F, typename ...A>
inline const T& ValXVector<T>::GetTruth(F&& Function, A && ...Args) const
{
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (Function(*it, std::forward<A>(Args)...))
            return (*it);
    }
    throw "Truth Not Found";
}

template<typename T>
template<typename F, typename ...A>
inline T& ValXVector<T>::GetTruth(F&& Function, A && ...Args)
{
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (Function(*it, std::forward<A>(Args)...))
            return (*it);
    }
    throw "Truth Not Found";
}

template<typename T>
template<typename F, typename ...A>
inline xvector<T> ValXVector<T>::GetTruths(F&& Function, A && ...Args) const
{
    xvector<T> RetVec;
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (Function(*it, std::forward<A>(Args)...))
            RetVec << (*it);
    }
    return RetVec;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline bool ValXVector<T>::Lacks(const T& item) const {
    return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
inline bool ValXVector<T>::Lacks(char const* item) const {
    return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
template<typename F, typename ...A>
inline bool ValXVector<T>::LacksTruth(F&& Function, A && ...Args) const
{
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (Function((*it), std::forward<A>(Args)...))
            return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------


template<typename T>
template<typename L>
xvector<T> ValXVector<T>::GetCommonItems(L& item) 
{
    std::sort(this->begin(), this->end());
    std::sort(item.begin(), item.end());

    xvector<T> vret(this->size() + item.size());
    set_intersection(this->begin(), this->end(), item.begin(), item.end(), vret.begin());
    return vret;
}

template<typename T>
template<typename L>
xvector<T> ValXVector<T>::GetCommonItems(L&& item) {
    return this->GetCommonItems(item);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline void ValXVector<T>::operator<<(const T& item)
{
    this->emplace_back(item);
}

template<typename T>
inline void ValXVector<T>::operator<<(T&& item){
    this->emplace_back(std::move(item));
}

template<typename T>
inline void ValXVector<T>::AddCharStrings(int strC, char** strV)
{
    for (int i = 0; i < strC; i++)
        this->push_back(T(strV[i]));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline T& ValXVector<T>::First(size_t Idx)
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Bounds";
    return this->operator[](Idx);
}

template<typename T>
inline const T& ValXVector<T>::First(size_t Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Bounds";
    return this->operator[](Idx);
}

template<typename T>
inline T& ValXVector<T>::Last(size_t Idx)
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Bounds";
    return this->operator[](this->size() - Idx - 1);
}


template<typename T>
inline const T& ValXVector<T>::Last(size_t Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Bounds";
    return this->operator[](this->size() - Idx - 1);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline std::pair<T, T> ValXVector<T>::GetPair() const
{
    return std::pair<E, E>(this->at(0), this->at(1));
}

template<typename T>
template<typename C>
inline xvector<C> ValXVector<T>::Convert() const
{
    xvector<C> ret;
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret << C(*it);
    return ret;
}


template<typename T>
template<typename C, typename F>
inline xvector<C> ValXVector<T>::Convert(F function) const
{
    xvector<C> ret;
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret << function(*it);
    return ret;
}

template<typename T>
template<typename N>
inline xvector<xvector<T>> ValXVector<T>::Split(N count) const
{

    xvector<xvector<T>> ret_vec;
    if (count < 2) {
        if (count == 1 && this->size() == 1) {
            ret_vec[0].reserve(this->size());
            for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++) {
                ret_vec[0].push_back(*it);
            }
        }
        else
            return ret_vec;
    }

    ret_vec.reserve(static_cast<size_t>(count) + 1);
    if (!this->size())
        return ret_vec;

    N reset = count;
    count = 0;
    const N new_size = static_cast<N>(this->size()) / reset;
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++) {
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
void ValXVector<T>::operator+=(const xvector<T>& other)
{
    this->insert(this->end(), other.begin(), other.end());
}

template<typename T>
xvector<T> ValXVector<T>::operator+(const xvector<T>& other) const 
{
    xvector<T> vret;
    vret.reserve(this->size());
    vret.insert(vret.end(),  this->begin(), this->end());
    vret += other;
    return vret;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline size_t ValXVector<T>::Size() const
{
    return this->size();
}

template<typename T>
inline void ValXVector<T>::Organize()
{
    std::multiset<T> set_arr;
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        set_arr.insert(*it);

    this->clear();
    this->reserve(set_arr.size());

    for (typename std::multiset<T>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        this->push_back(*it);
}

template<typename T>
inline void ValXVector<T>::RemoveDups()
{
    std::set<T> set_arr;
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        set_arr.insert(*it);

    this->clear();
    this->reserve(set_arr.size());

    for (typename std::set<T>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        this->push_back(*it);
}

// -----------------------------------------------------------------------------------------------

template<typename T>
template<typename F>
inline xvector<T>& ValXVector<T>::Sort(F func)
{
    std::sort(this->begin(), this->end(), func);
    return *reinterpret_cast<xvector<T>*>(this);
}

template<typename T>
inline xvector<T>& ValXVector<T>::Sort()
{
    std::sort(this->begin(), this->end());
    return *reinterpret_cast<xvector<T>*>(this);
}

template<typename T>
inline xvector<T>& ValXVector<T>::ReverseSort()
{
    std::sort(this->begin(), this->end(), std::greater<T>());
    return *reinterpret_cast<xvector<T>*>(this);
}

template<typename T>
inline xvector<T>& ValXVector<T>::ReverseIt()
{
    std::reverse(this->begin(), this->end());
    return *reinterpret_cast<xvector<T>*>(this);
}

template<typename T>
inline xvector<T*> ValXVector<T>::GetPtrs()
{
    xvector<T*> ret_vec;
    for (T& item : *this)
        ret_vec << &item;

    return ret_vec;
}

template<typename T>
template<typename F, typename... A>
inline void ValXVector<T>::Proc(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++) {
        if (function(*it, Args...))
            break;
    }
}

template<typename T>
template<typename F, typename... A>
inline void ValXVector<T>::ProcThread(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        Nexus<>::AddJobVal(function, *it, std::ref(Args)...);
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> ValXVector<T>::ForEach(F&& function, A&& ...Args) const
{
    xvector<N> vret;
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        vret.push_back(function(*it, Args...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> ValXVector<T>::ForEach(F&& function, A&& ...Args)
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        rmap.insert(function(*it, Args...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> ValXVector<T>::ForEach(F&& function, A&& ...Args)
{
    xvector<N> vret;
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        vret.push_back(function(*it, Args...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> ValXVector<T>::ForEach(F&& function, A&& ...Args) const
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        rmap.insert(function(*it, Args...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> ValXVector<T>::ForEachThread(F&& function, A&& ...Args)
{
    if constexpr (std::is_same_v<N, T>)
    {
        CheckRenewObj(VectorPool);
        for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
            VectorPool.AddJobVal(function, *it, std::ref(Args)...);

        return VectorPool.GetMoveAllIndices();
    }
    else
    {
        Nexus<N> LoNexus;
        for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
            LoNexus.AddJobVal(function, *it, std::ref(Args)...);

        return LoNexus.GetMoveAllIndices();
    }
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> ValXVector<T>::ForEachThread(F&& function, A&& ...Args) const
{
    Nexus<N> LoNexus;
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        LoNexus.AddJobVal(function, *it, std::ref(Args)...);

    return LoNexus.GetMoveAllIndices();
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> ValXVector<T>::ForEachThread(F&& function, A&& ...Args) const
{
    auto& MapPool = *RA::MakeShared<Nexus<std::pair<K, V>>>();
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        MapPool.AddJobVal(function, *it, std::ref(Args)...);

    return MapPool.GetMoveAllIndices();
}

template<typename T>
template <typename N, typename F, typename ...A>
inline void ValXVector<T>::StartTasks(F&& function, A&& ...Args)
{
    CheckRenewObj(VectorPool);
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        VectorPool.AddJobVal(function, *it, std::ref(Args)...);
}

template<typename T>
template<typename N>
inline typename std::enable_if<!std::is_same<N, void>::value, xvector<N>>::type ValXVector<T>::GetTasks() const
{
    if (!The.VectorPoolPtr)
        return std::vector<N>();
    return The.VectorPool->GetMoveAllIndices();
}

template<typename T>
inline bool ValXVector<T>::TasksCompleted() const
{
    if (!The.VectorPoolPtr)
        return true;
    return The.VectorPool->TasksCompleted();
}

// =============================================================================================================


template<typename T>
inline T ValXVector<T>::GetSum(size_t FnSkipIdx) const
{
    if (!Size())
        return 0;

    T LnModSize = 0;
    if (FnSkipIdx && Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 0;
    for (typename ValXVector<T>::const_iterator it = this->begin() + LnModSize; it != this->end(); it++) {
        num += *it;
    }
    return num;
}

template<typename T>
inline T ValXVector<T>::GetMul(size_t FnSkipIdx) const
{
    if (!Size())
        return 0;

    if (Size() == 1)
        return (*this)[0];

    T LnModSize = 0;
    if (Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 1;
    for (typename ValXVector<T>::const_iterator it = this->begin() + LnModSize; it != this->end(); it++) {
        num *= (*it);
    }
    return num;
}

template<typename T>
inline T ValXVector<T>::GetAvg(size_t FnSkipIdx) const
{
    return this->GetSum(FnSkipIdx) / (this->Size() - FnSkipIdx);
}

// =============================================================================================================

template<typename T>
inline T ValXVector<T>::Join(const T& str) const
{
    T ret;
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - str.size());
}

template<typename T>
T ValXVector<T>::Join(const char str) const
{
    T ret;
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - 1);
}

template<typename T>
T ValXVector<T>::Join(const char* str) const
{
    T ret;
    for (typename ValXVector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - strlen(str));
}

// =============================================================================================================


template<typename T>
bool ValXVector<T>::FullMatchOne(const re2::RE2& in_pattern) const {
    for (typename ValXVector<T>::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (RE2::FullMatch(*iter, in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool ValXVector<T>::FullMatchOne(const std::string& in_pattern) const {
    return this->FullMatchOne(in_pattern.c_str());
}

template<typename T>
bool ValXVector<T>::FullMatchOne(std::string&& in_pattern) const {
    return this->FullMatchOne(in_pattern.c_str());
}

template<typename T>
bool ValXVector<T>::FullMatchOne(char const* in_pattern) const {
    return this->FullMatchOne(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
bool ValXVector<T>::FullMatchAll(const re2::RE2& in_pattern) const {
    for (typename T::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (!RE2::FullMatch(*iter, in_pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool ValXVector<T>::FullMatchAll(const std::string& in_pattern) const {
    return this->FullMatchAll(in_pattern.c_str());
}

template<typename T>
bool ValXVector<T>::FullMatchAll(std::string&& in_pattern) const {
    return this->FullMatchAll(in_pattern.c_str());
}

template<typename T>
bool ValXVector<T>::FullMatchAll(char const* in_pattern) const {
    return this->FullMatchAll(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
bool ValXVector<T>::MatchOne(const re2::RE2& in_pattern) const {
    for (typename ValXVector<T>::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (RE2::PartialMatch(*iter, in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool ValXVector<T>::MatchOne(const std::string& in_pattern) const {
    return this->MatchOne(in_pattern.c_str());
}

template<typename T>
bool ValXVector<T>::MatchOne(std::string&& in_pattern) const {
    return this->MatchOne(in_pattern.c_str());
}

template<typename T>
bool ValXVector<T>::MatchOne(char const* in_pattern) const {
    return this->MatchOne(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
bool ValXVector<T>::MatchAll(const re2::RE2& in_pattern) const {
    for (typename T::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (!RE2::PartialMatch(*iter, in_pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool ValXVector<T>::MatchAll(const std::string& in_pattern) const {
    return this->MatchAll(in_pattern.c_str());
}

template<typename T>
bool ValXVector<T>::MatchAll(std::string&& in_pattern) const {
    return this->MatchAll(in_pattern.c_str());
}

template<typename T>
bool ValXVector<T>::MatchAll(char const* in_pattern) const {
    return this->MatchAll(re2::RE2(in_pattern));
}
// =============================================================================================================

template<typename T>
inline xvector<T> ValXVector<T>::Take(const re2::RE2& in_pattern) const
{
    xvector<T> ret_vec;
    ret_vec.reserve(this->size() + 1);
    for (size_t i = 0; i < this->size(); i++) {
        if ((RE2::PartialMatch((*this)[i], in_pattern)))
            ret_vec.push_back((*this)[i]);
    }
    return ret_vec;
}

template<typename T>
inline xvector<T> ValXVector<T>::Take(const std::string& in_pattern) const
{
    return this->Take(in_pattern.c_str());
}

template<typename T>
inline xvector<T> ValXVector<T>::Take(std::string&& in_pattern) const
{
    return this->Take(in_pattern.c_str());
}

template<typename T>
inline xvector<T> ValXVector<T>::Take(char const* in_pattern) const
{
    return this->Take(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
inline xvector<T> ValXVector<T>::Remove(const re2::RE2& in_pattern) const
{
    xvector<T> ret_vec;
    ret_vec.reserve(this->size() + 1);
    for (size_t i = 0; i < this->size(); i++) {
        if (!(RE2::PartialMatch((*this)[i].c_str(), in_pattern)))
            ret_vec.push_back((*this)[i]);
    }
    return ret_vec;
}

template<typename T>
inline xvector<T> ValXVector<T>::Remove(const std::string& in_pattern) const
{
    return this->Remove(in_pattern.c_str());
}

template<typename T>
inline xvector<T> ValXVector<T>::Remove(std::string&& in_pattern) const
{
    return this->Remove(in_pattern.c_str());
}

template<typename T>
inline xvector<T> ValXVector<T>::Remove(char const* in_pattern) const
{
    return this->Remove(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
inline xvector<T> ValXVector<T>::SubAll(const re2::RE2& in_pattern, const std::string& replacement) const
{
    xvector<T> ret_vec = *this;
    for (typename ValXVector<T>::iterator iter = ret_vec.begin(); iter != ret_vec.end(); iter++)
        RE2::GlobalReplace(&*iter, in_pattern, replacement.c_str());
    return ret_vec;
}

template<typename T>
inline xvector<T> ValXVector<T>::SubAll(const std::string& in_pattern, const std::string& replacement) const
{
    return this->SubAll(in_pattern.c_str(), replacement);
}

template<typename T>
inline xvector<T> ValXVector<T>::SubAll(std::string&& in_pattern, std::string&& replacement) const
{
    return this->SubAll(in_pattern.c_str(), replacement);
}

template<typename T>
inline xvector<T> ValXVector<T>::SubAll(char const* in_pattern, char const* replacement) const
{
    return this->SubAll(re2::RE2(in_pattern), replacement);
}

// =============================================================================================================

template<typename T>
xvector<T> ValXVector<T>::operator()(long double x, long double y, long double z, const char removal_method) const {

    size_t m_size = this->size();
    xvector<T> n_arr;
    n_arr.reserve(m_size + 4);

    double n_arr_size = static_cast<double>(m_size) - 1;

    if (z >= 0) {

        if (x < 0) { x += n_arr_size; }

        if (!y) { y = n_arr_size; }
        else if (y < 0) { y += n_arr_size; }
        ++y;

        if (x > y) { return n_arr; }

        typename ValXVector<T>::const_iterator iter = this->begin();
        typename ValXVector<T>::const_iterator stop = this->begin() + static_cast<size_t>(y);

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
        
        typename ValXVector<T>::const_reverse_iterator iter = this->rend() - static_cast<size_t>(x) - 1;
        typename ValXVector<T>::const_reverse_iterator stop = this->rend() - static_cast<size_t>(y);

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
