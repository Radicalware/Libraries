#pragma once
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

template<typename T> class SPtrXVector;

template<typename T>
class SPtrXVector<xp<T>> : public BaseXVector<xp<T>>
{    
public:
    using BaseXVector<xp<T>>::BaseXVector;
    using BaseXVector<xp<T>>::operator=;
    using E = std::remove_const<xp<T>>::type; // E for Erratic

    inline T& At(const size_t Idx);
    inline const T& At(const size_t Idx) const;

    template<typename P = T> 
    inline bool Has(const P* item) const;
    inline bool Has(const T& item) const;
    inline bool Has(T&& item) const;
    inline bool Has(char const* item) const;

    inline bool Lacks(const T& item) const;
    inline bool Lacks(T&& item) const;
    inline bool Lacks(char const* item) const;

    template<typename L = xvector<xp<T>>>
    inline xvector<xp<T>> GetCommonItems(L& item);
    template<typename L = xvector<xp<T>>>
    inline xvector<xp<T>> GetCommonItems(L&& item);

    template<typename ...R>
    inline void Emplace(R&& ... Args);
    inline void operator<<(const xp<T>&  item);
    inline void operator<<(      xp<T>&& item);

    T& First(size_t value = 0);
    const T& First(size_t value = 0) const;

    T& Last(size_t value = 0);
    const T& Last(size_t value = 0) const;

    template<typename C>
    inline xvector<C> Convert() const;

    template<typename C, typename F>
    inline xvector<C> Convert(F function) const;

    template<typename N = unsigned int>
    inline xvector<xvector<xp<T>>> Split(N count = 1) const;

    inline void operator+=(const xvector<xp<T>>& other);
    inline xvector<xp<T>> operator+(const xvector<xp<T>>& other) const;

    size_t Size() const;

    inline void Organize();
    inline void RemoveDups();

    template<typename F>
    inline xvector<xp<T>>& Sort(F func);
    inline xvector<xp<T>>& Sort();
    inline xvector<xp<T>>& ReverseSort();
    inline xvector<xp<T>>& ReverseIt();

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

    inline xvector<xp<T>> Take(const re2::RE2& in_pattern) const;
    inline xvector<xp<T>> Take(const std::string& in_pattern) const;
    inline xvector<xp<T>> Take(std::string&& in_pattern) const;
    inline xvector<xp<T>> Take(char const* in_pattern) const;

    inline xvector<xp<T>> Remove(const re2::RE2& in_pattern) const;
    inline xvector<xp<T>> Remove(const std::string& in_pattern) const;
    inline xvector<xp<T>> Remove(std::string&& in_pattern) const;
    inline xvector<xp<T>> Remove(char const* in_pattern) const;

    inline xvector<xp<T>> SubAll(const re2::RE2& in_pattern, const std::string& replacement) const;
    inline xvector<xp<T>> SubAll(const std::string& in_pattern, const std::string& replacement) const;
    inline xvector<xp<T>> SubAll(std::string&& in_pattern, std::string&& replacement) const;
    inline xvector<xp<T>> SubAll(char const* in_pattern, char const* replacement) const;
    
    // double was chose to hold long signed and unsigned values
    inline xvector<xp<T>> operator()(long double x = 0, long double y = 0, long double z = 0, const char removal_method = 's') const;
    // s = slice perserves values you land on 
    // d = dice  removes values you land on
    // s/d only makes a difference if you modify the 'z' value

    // =================================== DESIGNED FOR STRING BASED VECTORS ===================================
};
// =============================================================================================================

template<typename T>
inline T& SPtrXVector<xp<T>>::At(const size_t Idx)
{
    if (Idx >= Size())
        throw "Index Out Of Range";
    return (*this)[Idx].Get();
}

template<typename T>
inline const T& SPtrXVector<xp<T>>::At(const size_t Idx) const
{
    if (Idx >= Size())
        throw "Index Out Of Range";
    return (*this)[Idx].Get();
}

template<typename T>
template<typename P>
inline bool SPtrXVector<xp<T>>::Has(const P* item) const
{
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (**it == *item)
            return true;
    }
    return false;
}

template<typename T>
bool SPtrXVector<xp<T>>::Has(const T& item) const {
    return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool SPtrXVector<xp<T>>::Has(T&& item) const {
    return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool SPtrXVector<xp<T>>::Has(char const* item) const {
    return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
bool SPtrXVector<xp<T>>::Lacks(T&& item) const {
    return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool SPtrXVector<xp<T>>::Lacks(const T& item) const {
    return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool SPtrXVector<xp<T>>::Lacks(char const* item) const {
    return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename L>
xvector<xp<T>> SPtrXVector<xp<T>>::GetCommonItems(L& item) 
{
    std::sort(this->begin(), this->end());
    std::sort(item.begin(), item.end());

    xvector<xp<T>> vret(this->size() + item.size());
    set_intersection(this->begin(), this->end(), item.begin(), item.end(), vret.begin());
    return vret;
}

template<typename T>
template<typename L>
xvector<xp<T>> SPtrXVector<xp<T>>::GetCommonItems(L&& item) {
    return this->GetCommonItems(item);
}

template<typename T>
template<typename ...R>
inline void SPtrXVector<xp<T>>::Emplace(R && ...Args)
{
    this->emplace_back(RA::MakeShared<T>(std::forward<R>(Args)...));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline void SPtrXVector<xp<T>>::operator<<(const xp<T>& item)
{
    this->emplace_back(item);
}

template<typename T>
inline void SPtrXVector<xp<T>>::operator<<(xp<T>&& item){
    this->emplace_back(std::move(item));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline T& SPtrXVector<xp<T>>::First(size_t value)
{
    return this->operator[](value).Get();
}

template<typename T>
inline const T& SPtrXVector<xp<T>>::First(size_t value) const
{
    return this->operator[](value).Get();
}

template<typename T>
inline T& SPtrXVector<xp<T>>::Last(size_t value) {
    return this->operator[](this->size() - value - 1).Get();
}


template<typename T>
inline const T& SPtrXVector<xp<T>>::Last(size_t value) const {
    return this->operator[](this->size() - value - 1).Get();
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename C>
inline xvector<C> SPtrXVector<xp<T>>::Convert() const
{
    xvector<C> ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        ret << C((*it).Get());
    return ret;
}


template<typename T>
template<typename C, typename F>
inline xvector<C> SPtrXVector<xp<T>>::Convert(F function) const
{
    xvector<C> ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        ret << function((*it).Get());
    return ret;
}

template<typename T>
template<typename N>
inline xvector<xvector<xp<T>>> SPtrXVector<xp<T>>::Split(N count) const
{
    xvector<xvector<xp<T>>> ret_vec;
    if (!this->size())
        return ret_vec;

    if (count < 2) {
        if (count == 1 && this->size() == 1) {
            ret_vec[0].reserve(this->size());
            for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++) {
                ret_vec[0].push_back(*it);
            }
        }
        else
            return ret_vec;
    }

    ret_vec.reserve(static_cast<size_t>(count) + 1);

    N reset = count;
    count = 0;
    const N new_size = static_cast<N>(this->size()) / reset;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (count == 0) {
            count = reset;
            ret_vec.push_back(xvector<xp<T>>({ *it })); // create new xvec and add first el
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
void SPtrXVector<xp<T>>::operator+=(const xvector<xp<T>>& other)
{
    this->insert(this->end(), other.begin(), other.end());
}

template<typename T>
xvector<xp<T>> SPtrXVector<xp<T>>::operator+(const xvector<xp<T>>& other) const 
{
    xvector<xp<T>> vret;
    vret.reserve(this->size());
    vret.insert(vret.end(),  this->begin(), this->end());
    vret += other;
    return vret;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline size_t SPtrXVector<xp<T>>::Size() const
{
    return this->size();
}

template<typename T>
inline void SPtrXVector<xp<T>>::Organize()
{
    std::multiset<xp<T>> set_arr;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        set_arr.insert(*it);

    this->clear();
    this->reserve(set_arr.size());

    for (typename std::multiset<xp<T>>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        this->push_back(*it);
}

template<typename T>
inline void SPtrXVector<xp<T>>::RemoveDups()
{
    std::set<xp<T>> set_arr;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        set_arr.insert(*it);

    this->clear();
    this->reserve(set_arr.size());

    for (typename std::set<xp<T>>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        this->push_back(*it);
}

// -----------------------------------------------------------------------------------------------

template<typename T>
template<typename F>
inline xvector<xp<T>>& SPtrXVector<xp<T>>::Sort(F func)
{
    std::sort(this->begin(), this->end(), func);
    return *reinterpret_cast<xvector<xp<T>>*>(this);
}

template<typename T>
inline xvector<xp<T>>& SPtrXVector<xp<T>>::Sort()
{
    std::sort(this->begin(), this->end());
    return *reinterpret_cast<xvector<xp<T>>*>(this);
}

template<typename T>
inline xvector<xp<T>>& SPtrXVector<xp<T>>::ReverseSort()
{
    std::sort(this->begin(), this->end(), std::greater<xp<T>>());
    return *reinterpret_cast<xvector<xp<T>>*>(this);
}

template<typename T>
inline xvector<xp<T>>& SPtrXVector<xp<T>>::ReverseIt()
{
    std::reverse(this->begin(), this->end());
    return *reinterpret_cast<xvector<xp<T>>*>(this);
}

template<typename T>
template<typename F, typename... A>
inline void SPtrXVector<xp<T>>::Proc(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++) {
        if (function((*it).Get(), Args...))
            break;
    }
}

template<typename T>
template<typename F, typename... A>
inline void SPtrXVector<xp<T>>::ProcThread(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        Nexus<>::AddJobVal(function, (*it).Get(), std::ref(Args)...);
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> SPtrXVector<xp<T>>::ForEach(F&& function, A&& ...Args) const
{
    xvector<N> vret;
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        vret.push_back(function((*it).Get(), Args...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> SPtrXVector<xp<T>>::ForEach(F&& function, A&& ...Args)
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        rmap.insert(function((*it).Get(), Args...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> SPtrXVector<xp<T>>::ForEach(F&& function, A&& ...Args)
{
    xvector<N> vret;
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        vret.push_back(function((*it).Get(), Args...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> SPtrXVector<xp<T>>::ForEach(F&& function, A&& ...Args) const
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        rmap.insert(function((*it).Get(), Args...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> SPtrXVector<xp<T>>::ForEachThread(F&& function, A&& ...Args)
{
    if constexpr (std::is_same_v<N, T>)
    {
        CheckRenewObject(VectorPool);
        for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
            VectorPool.AddJobVal(function, (*it).Get(), std::ref(Args)...);

        return VectorPool.GetMoveAllIndices();
    }
    else
    {
        Nexus<N> LoNexus;
        for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
            LoNexus.AddJobVal(function, (*it).Get(), std::ref(Args)...);

        return LoNexus.GetMoveAllIndices();
    }
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> SPtrXVector<xp<T>>::ForEachThread(F&& function, A&& ...Args) const
{
    Nexus<N> LoNexus;
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        LoNexus.AddJobVal(function, (*it).Get(), std::ref(Args)...);

    return LoNexus.GetMoveAllIndices();
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> SPtrXVector<xp<T>>::ForEachThread(F&& function, A&& ...Args) const
{
    auto& MapPool = *RA::MakeShared<Nexus<std::pair<K, V>>>();
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        MapPool.AddJobVal(function, (*it).Get(), std::ref(Args)...);

    return MapPool.GetMoveAllIndices();
}

template<typename T>
template <typename N, typename F, typename ...A>
inline void SPtrXVector<xp<T>>::StartTasks(F&& function, A&& ...Args)
{
    CheckRenewObject(VectorPool);
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        VectorPool.AddJobVal(function, (*it).Get(), std::ref(Args)...);
}

template<typename T>
template<typename N>
inline typename std::enable_if<!std::is_same<N, void>::value, xvector<N>>::type SPtrXVector<xp<T>>::GetTasks() const
{
    if (!The.VectorPoolPtr)
        return std::vector<N>();
    return The.VectorPool->GetMoveAllIndices();
}

template<typename T>
inline bool SPtrXVector<xp<T>>::TasksCompleted() const
{
    if (!The.VectorPoolPtr)
        return true;
    return The.VectorPool->TasksCompleted();
}

// =============================================================================================================


template<typename T>
inline T SPtrXVector<xp<T>>::GetSum(size_t FnSkipIdx) const
{
    if (!Size())
        return 0;

    T LnModSize = 0;
    if (FnSkipIdx && Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 0;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin() + LnModSize; it != this->end(); it++) {
        num += (*it).Get();
    }
    return num;
}

template<typename T>
inline T SPtrXVector<xp<T>>::GetMul(size_t FnSkipIdx) const
{
    if (!Size())
        return 0;

    if (Size() == 1)
        return (*this)[0];

    T LnModSize = 0;
    if (Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 1;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin() + LnModSize; it != this->end(); it++) {
        num *= (*it).Get();
    }
    return num;
}

template<typename T>
inline T SPtrXVector<xp<T>>::GetAvg(size_t FnSkipIdx) const
{
    return this->GetSum(FnSkipIdx) / (this->Size() - FnSkipIdx);
}

// =============================================================================================================

template<typename T>
inline T SPtrXVector<xp<T>>::Join(const T& str) const
{
    T ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += (*it).Get() + str;

    return ret.substr(0, ret.length() - str.size());
}

template<typename T>
T SPtrXVector<xp<T>>::Join(const char str) const
{
    T ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += (*it).Get() + str;

    return ret.substr(0, ret.length() - 1);
}

template<typename T>
T SPtrXVector<xp<T>>::Join(const char* str) const
{
    T ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += (*it).Get() + str;

    return ret.substr(0, ret.length() - strlen(str));
}

// =============================================================================================================


template<typename T>
bool SPtrXVector<xp<T>>::FullMatchOne(const re2::RE2& in_pattern) const {
    for (typename SPtrXVector<xp<T>>::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (RE2::FullMatch((*iter).Get(), in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool SPtrXVector<xp<T>>::FullMatchOne(const std::string& in_pattern) const {
    return this->FullMatchOne(in_pattern.c_str());
}

template<typename T>
bool SPtrXVector<xp<T>>::FullMatchOne(std::string&& in_pattern) const {
    return this->FullMatchOne(in_pattern.c_str());
}

template<typename T>
bool SPtrXVector<xp<T>>::FullMatchOne(char const* in_pattern) const {
    return this->FullMatchOne(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
bool SPtrXVector<xp<T>>::FullMatchAll(const re2::RE2& in_pattern) const {
    for (typename T::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (!RE2::FullMatch((*iter).Get(), in_pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool SPtrXVector<xp<T>>::FullMatchAll(const std::string& in_pattern) const {
    return this->FullMatchAll(in_pattern.c_str());
}

template<typename T>
bool SPtrXVector<xp<T>>::FullMatchAll(std::string&& in_pattern) const {
    return this->FullMatchAll(in_pattern.c_str());
}

template<typename T>
bool SPtrXVector<xp<T>>::FullMatchAll(char const* in_pattern) const {
    return this->FullMatchAll(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
bool SPtrXVector<xp<T>>::MatchOne(const re2::RE2& in_pattern) const {
    for (typename SPtrXVector<xp<T>>::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (RE2::PartialMatch((*iter).Get(), in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool SPtrXVector<xp<T>>::MatchOne(const std::string& in_pattern) const {
    return this->MatchOne(in_pattern.c_str());
}

template<typename T>
bool SPtrXVector<xp<T>>::MatchOne(std::string&& in_pattern) const {
    return this->MatchOne(in_pattern.c_str());
}

template<typename T>
bool SPtrXVector<xp<T>>::MatchOne(char const* in_pattern) const {
    return this->MatchOne(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
bool SPtrXVector<xp<T>>::MatchAll(const re2::RE2& in_pattern) const {
    for (typename T::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (!RE2::PartialMatch((*iter).Get(), in_pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool SPtrXVector<xp<T>>::MatchAll(const std::string& in_pattern) const {
    return this->MatchAll(in_pattern.c_str());
}

template<typename T>
bool SPtrXVector<xp<T>>::MatchAll(std::string&& in_pattern) const {
    return this->MatchAll(in_pattern.c_str());
}

template<typename T>
bool SPtrXVector<xp<T>>::MatchAll(char const* in_pattern) const {
    return this->MatchAll(re2::RE2(in_pattern));
}
// =============================================================================================================

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Take(const re2::RE2& in_pattern) const
{
    xvector<xp<T>> ret_vec;
    ret_vec.reserve(this->size() + 1);
    for (size_t i = 0; i < this->size(); i++) {
        if ((RE2::PartialMatch((*this)[i], in_pattern)))
            ret_vec.push_back((*this)[i]);
    }
    return ret_vec;
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Take(const std::string& in_pattern) const
{
    return this->Take(in_pattern.c_str());
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Take(std::string&& in_pattern) const
{
    return this->Take(in_pattern.c_str());
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Take(char const* in_pattern) const
{
    return this->Take(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Remove(const re2::RE2& in_pattern) const
{
    xvector<xp<T>> ret_vec;
    ret_vec.reserve(this->size() + 1);
    for (size_t i = 0; i < this->size(); i++) {
        if (!(RE2::PartialMatch((*this)[i].c_str(), in_pattern)))
            ret_vec.push_back((*this)[i]);
    }
    return ret_vec;
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Remove(const std::string& in_pattern) const
{
    return this->Remove(in_pattern.c_str());
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Remove(std::string&& in_pattern) const
{
    return this->Remove(in_pattern.c_str());
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Remove(char const* in_pattern) const
{
    return this->Remove(re2::RE2(in_pattern));
}

// =============================================================================================================

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::SubAll(const re2::RE2& in_pattern, const std::string& replacement) const
{
    xvector<xp<T>> ret_vec = *this;
    for (typename SPtrXVector<xp<T>>::iterator iter = ret_vec.begin(); iter != ret_vec.end(); iter++)
        RE2::GlobalReplace(&(*iter).Get(), in_pattern, replacement.c_str());
    return ret_vec;
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::SubAll(const std::string& in_pattern, const std::string& replacement) const
{
    return this->SubAll(in_pattern.c_str(), replacement);
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::SubAll(std::string&& in_pattern, std::string&& replacement) const
{
    return this->SubAll(in_pattern.c_str(), replacement);
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::SubAll(char const* in_pattern, char const* replacement) const
{
    return this->SubAll(re2::RE2(in_pattern), replacement);
}

// =============================================================================================================

template<typename T>
xvector<xp<T>> SPtrXVector<xp<T>>::operator()(long double x, long double y, long double z, const char removal_method) const {

    size_t m_size = this->size();
    xvector<xp<T>> n_arr;
    n_arr.reserve(m_size + 4);

    double n_arr_size = static_cast<double>(m_size) - 1;

    if (z >= 0) {

        if (x < 0) { x += n_arr_size; }

        if (!y) { y = n_arr_size; }
        else if (y < 0) { y += n_arr_size; }
        ++y;

        if (x > y) { return n_arr; }

        typename SPtrXVector<xp<T>>::const_iterator iter = this->begin();
        typename SPtrXVector<xp<T>>::const_iterator stop = this->begin() + static_cast<size_t>(y);

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
        
        typename SPtrXVector<xp<T>>::const_reverse_iterator iter = this->rend() - static_cast<size_t>(x) - 1;
        typename SPtrXVector<xp<T>>::const_reverse_iterator stop = this->rend() - static_cast<size_t>(y);

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
