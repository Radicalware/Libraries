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


#include "XvectorTypes.h"
#include "BaseXVector.h"
#include <regex>

template<typename T>
class PtrXVector<T*> : public BaseXVector<T*>
{
protected:
    xp<Nexus<T>> MoNexusPtr = nullptr;
public:
    using BaseXVector<T*>::BaseXVector;
    using BaseXVector<T*>::operator=;
    using E = typename std::remove_const<T>::type; // E for Erratic

    RIN T& At(const xint Idx);
    RIN const T& At(const xint Idx) const;

    RIN bool Has(const T& item) const;
    RIN bool Has(char const* item) const;
    template<typename F, typename ...A>
    RIN bool HasTruth(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    RIN const T& GetTruth(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    RIN T& GetTruth(F&& Function, A&& ...Args);
    template<typename F, typename ...A>
    RIN xvector<const T*> GetTruths(F&& Function, A&& ...Args) const;

    RIN bool Lacks(const T& item) const;
    RIN bool Lacks(char const* item) const;
    template<typename F, typename ...A>
    RIN bool LacksTruth(F&& Function, A&& ...Args) const;


    RIN const T* RawPtr(xint Idx) const;
    RIN T* RawPtr(xint Idx);
    RIN const T& operator[](xint Idx) const;
    RIN T& operator[](xint Idx);

    const T& First(xint Idx = 0) const;
    T& First(xint Idx = 0);

    const T& Last(xint Idx = 0) const;
    T& Last(xint Idx = 0);

    RIN std::pair<T, T> GetPair() const;

    template<typename I>
    RIN xvector<I> Convert() const;

    template<typename I, typename F>
    RIN xvector<I> Convert(F function) const;
    
    template<typename N = unsigned int>
    xvector<xvector<T*>> Split(N count) const;


    RIN void operator<<(T* item);

    RIN void operator+=(const xvector<T*>& other);
    RIN xvector<T*> operator+(const xvector<T*>& other) const;

    RIN xint Size() const;

    RIN void Organize();
    RIN void RemoveDups();

    template<typename F>
    RIN xvector<T*> Sort(F func);

    RIN xvector<T> GetVals() const;
    RIN T* at(const xint idx) const;

    template<typename F, typename... A>
    RIN void Proc(F&& function, A&& ...Args);
    template<typename F, typename... A>
    RIN void ProcThread(F&& function, A&& ...Args);

    // foreach non-const
    template<typename N = E, typename F, typename ...A>             RIN xvector<N> ForEach(F&& function, A&& ...Args);
    template<typename K, typename V, typename F, typename ...A>     RIN std::unordered_map<K, V> ForEach(F&& function, A&& ...Args);
    // foreach const
    template<typename N = E, typename F, typename ...A>             RIN xvector<N> ForEach(F&& function, A&& ...Args) const;
    template<typename K, typename V, typename F, typename ...A>     RIN std::unordered_map<K, V> ForEach(F&& function, A&& ...Args) const;
    // multi-threaded foreach
    template<typename N = E, typename F, typename... A>             RIN xvector<xp<N>> ForEachThread(F&& function, A&& ...Args);
    template<typename N = E, typename F, typename... A>             RIN xvector<xp<N>> ForEachThread(F&& function, A&& ...Args) const;
    template<typename K, typename V, typename F, typename ...A>     RIN std::unordered_map<K, V> ForEachThread(F&& function, A&& ...Args) const;

    template<typename N = E, typename F, typename... A>             RIN void StartTasks(F&& function, A&& ...Args);
    template<typename N = E>                                        RIN xvector<xp<N>> GetTasks() const;
                                                                    RIN bool TasksCompleted() const;

    // =================================== DESIGNED FOR NUMERIC BASED VECTORS ===================================

    RIN T GetSum(xint FnSkipIdx = 0) const;
    RIN T GetMul(xint FnSkipIdx = 0) const;
    RIN T GetAvg(xint FnSkipIdx = 0) const;

    // =================================== DESIGNED FOR STRING  BASED VECTORS ===================================

    RIN T Join(const T& str = "") const;
    RIN T Join(const char str) const;
    RIN T Join(const char* str) const;

#ifndef UsingNVCC
    RIN bool FullMatchOne(const re2::RE2& in_pattern) const;
    RIN bool FullMatchAll(const re2::RE2& in_pattern) const;
    RIN bool MatchOne(const re2::RE2& in_pattern) const;
    RIN bool MatchAll(const re2::RE2& in_pattern) const;
    RIN xvector<T*> Take(const re2::RE2& in_pattern) const;
    RIN xvector<T*> Remove(const re2::RE2& in_pattern) const;
    RIN xvector<T>  SubAll(const re2::RE2& in_pattern, const std::string& replacement) const;
#endif // !UsingNVCC


    RIN bool FullMatchOne(const std::string& in_pattern) const;
    RIN bool FullMatchOne(char const* in_pattern) const;

    RIN bool FullMatchAll(const std::string& in_pattern) const;
    RIN bool FullMatchAll(char const* in_pattern) const;

    RIN bool MatchOne(const std::string& in_pattern) const;
    RIN bool MatchOne(char const* in_pattern) const;

    RIN bool MatchAll(const std::string& in_pattern) const;
    RIN bool MatchAll(char const* in_pattern) const;

    RIN xvector<T*> Take(const std::string& in_pattern) const;
    RIN xvector<T*> Take(char const* in_pattern) const;

    RIN xvector<T*> Remove(const std::string& in_pattern) const;
    RIN xvector<T*> Remove(char const* in_pattern) const;

    RIN xvector<T> SubAll(const std::string& in_pattern, const std::string& replacement) const;
    RIN xvector<T> SubAll(char const* in_pattern, char const* replacement) const;

    // double was chose to hold long signed and unsigned values
    RIN xvector<T*> operator()(long double x = 0, long double y = 0, long double z = 0, const char removal_method = 's') const;
    // s = slice perserves values you land on 
    // d = dice  removes values you land on
    // s/d only makes a difference if you modify the 'z' value
    
    // =================================== DESIGNED FOR STRING BASED VECTORS ===================================

    RIN void SetDeleteAllOnExit(const bool Truth) { MbDeleteAllOnExit = Truth; }

protected:
    bool MbDeleteAllOnExit = false;
};
// =============================================================================================================


template<typename T>
RIN T& PtrXVector<T*>::At(const xint Idx)
{
    if (Idx >= Size())
        throw "Index Out Of Range";
    return *(*this)[Idx];
}

template<typename T>
RIN const T& PtrXVector<T*>::At(const xint Idx) const
{
    if (Idx >= Size())
        throw "Index Out Of Range";
    return *(*this)[Idx];
}

template<typename T>
RIN bool PtrXVector<T*>::Has(const T& item)  const
{
    for (auto* el : *this) {
        if (*el == item)
            return true;
    }
    return false;
}

template<typename T>
RIN bool PtrXVector<T*>::Has(char const* item)  const
{
    for (auto* el : *this) {
        if (*el == item)
            return true;
    }
    return false;
}

template<typename T>
template<typename F, typename ...A>
RIN bool PtrXVector<T*>::HasTruth(F&& Function, A&& ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((**it), std::forward<A>(Args)...))
            return true;
    }
    return false;
}

template<typename T>
template<typename F, typename ...A>
RIN const T& PtrXVector<T*>::GetTruth(F&& Function, A && ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function(**it, std::forward<A>(Args)...))
            return **it;
    }
    throw "Truth Not Found";
}

template<typename T>
template<typename F, typename ...A>
RIN T& PtrXVector<T*>::GetTruth(F&& Function, A && ...Args)
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function(**it, std::forward<A>(Args)...))
            return **it;
    }
    throw "Truth Not Found";
}

template<typename T>
template<typename F, typename ...A>
RIN xvector<const T*> PtrXVector<T*>::GetTruths(F&& Function, A && ...Args) const
{
    xvector<T*> RetVec;
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function(**it, std::forward<A>(Args)...))
            RetVec << **it;
    }
    return RetVec;
}

template<typename T>
RIN bool PtrXVector<T*>::Lacks(const T& item) const {
    return !(bool(std::find(The.begin(), The.end(), &item) != The.end()));
}

template<typename T>
RIN bool PtrXVector<T*>::Lacks(char const* item) const {
    return !(bool(std::find(The.begin(), The.end(), &item) != The.end()));
}

template<typename T>
template<typename F, typename ...A>
RIN bool PtrXVector<T*>::LacksTruth(F&& Function, A&& ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((**it), std::forward<A>(Args)...))
            return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
RIN const T* PtrXVector<T*>::RawPtr(xint Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Index out of range";
    return BaseXVector<T*>::operator[](Idx);
}

template<typename T>
RIN T* PtrXVector<T*>::RawPtr(xint Idx)
{
    if (!The.HasIndex(Idx))
        throw "Index out of range";
    return BaseXVector<T*>::operator[](Idx);
}

template<typename T>
RIN const T& PtrXVector<T*>::operator[](xint Idx) const
{
    const T* RetPtr = The.RawPtr(Idx);
    if (!RetPtr)
        throw "Nullptr Hit";
    return *RetPtr;
}

template<typename T>
RIN T& PtrXVector<T*>::operator[](xint Idx)
{
    T* RetPtr = The.RawPtr(Idx);
    if (!RetPtr)
        throw "Nullptr Hit";
    return *RetPtr;
}

template<typename T>
RIN const T& PtrXVector<T*>::First(xint Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Range";
    return The.operator[](Idx);
}

template<typename T>
RIN T& PtrXVector<T*>::First(xint Idx)
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Range";
    return The.operator[](Idx);
}

template<typename T>
RIN const T& PtrXVector<T*>::Last(xint Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Range";
    return The.operator[](The.size() - Idx - 1);
}

template<typename T>
RIN T& PtrXVector<T*>::Last(xint Idx)
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Range";
    return The.operator[](The.size() - Idx - 1);
}

template<typename T>
RIN std::pair<T, T> PtrXVector<T*>::GetPair() const
{
    return std::pair<E*, E*>(The.at(0), The.at(1));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename I>
RIN xvector<I> PtrXVector<T*>::Convert() const
{
    xvector<I> ret;
    for (typename xvector<T*>::const_iterator it = The.begin(); it != The.end(); it++)
        ret << I(**it);
    return ret;
}

template<typename T>
template<typename I, typename F>
RIN xvector<I> PtrXVector<T*>::Convert(F function) const
{
    xvector<I> ret;
    for (typename xvector<T*>::const_iterator it = The.begin(); it != The.end(); it++)
        ret << function(*it);
    return ret;
}

template<typename T>
template<typename N>
RIN xvector<xvector<T*>> PtrXVector<T*>::Split(N count) const
{
    if (count < 2)
        return xvector<xvector<T*>>{ *this };

    xvector<xvector<T*>> RetVec;
    RetVec.reserve(static_cast<xint>(count) + 1);
    if (!The.size())
        return RetVec;

    N reset = count;
    count = 0;
    const N new_size = static_cast<N>(The.size()) / reset;
    for (typename xvector<T*>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (count == 0) {
            count = reset;
            RetVec.push_back(xvector<T*>({ *it })); // create new xvec and add first el
            RetVec[RetVec.size() - 1].reserve(static_cast<xint>(new_size));
        }
        else {
            RetVec[RetVec.size() - 1] << *it;
        }
        count--;
    }
    return RetVec;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
RIN void PtrXVector<T*>::operator<<(T* item)
{
    The.emplace_back(item);
}

template<typename T>
void PtrXVector<T*>::operator+=(const xvector<T*>& other)
{
    The.insert(The.end(), other.begin(), other.end());
}

template<typename T>
xvector<T*> PtrXVector<T*>::operator+(const xvector<T*>& other) const {
    xint sz = The.size();
    xvector<T*> vret = *this;
    vret += other;
    return vret;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
RIN xint PtrXVector<T*>::Size() const
{
    return The.size();
}

template<typename T>
RIN void PtrXVector<T*>::Organize()
{
    std::multiset<T*> set_arr;
    for (typename xvector<T*>::const_iterator it = The.begin(); it != The.end(); it++)
        set_arr.insert(*it);

    The.clear();
    The.reserve(set_arr.size());

    for (typename std::multiset<T*>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        The.push_back(*it);
}

template<typename T>
RIN void PtrXVector<T*>::RemoveDups()
{
    std::set<T*> set_arr;
    for (typename xvector<T*>::const_iterator it = The.begin(); it != The.end(); it++)
        set_arr.insert(*it);

    The.clear();
    The.reserve(set_arr.size());

    for (typename std::set<T*>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        The.push_back(*it);
}

// ------------------------------------------------------------------------------------------------


template<typename T>
template<typename F>
RIN xvector<T*> PtrXVector<T*>::Sort(F func)
{
    std::sort(The.begin(), The.end(), func);
    return *this;
}

template<typename T>
RIN xvector<T> PtrXVector<T*>::GetVals() const
{
    xvector<E> arr;
    arr.reserve(The.size() + 1);
    for (typename xvector<T*>::const_iterator it = The.begin(); it != The.end(); it++)
        arr.push_back(**it);
    return arr;
}

template<typename T>
RIN T* PtrXVector<T*>::at(const xint idx) const
{
    if (idx >= The.size())
        throw std::out_of_range(std::string("Index [") + std::to_string(idx) + "] is out of range!");
    else
        return (*this)[idx];
}

template<typename T>
template<typename F, typename... A>
RIN void PtrXVector<T*>::Proc(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++) {
        if (function(**it, std::forward<A>(Args)...))
            break;
    }
}

template<typename T>
template<typename F, typename... A>
RIN void PtrXVector<T*>::ProcThread(F&& function, A&& ...Args)
{
    for (typename xvector<E*>::iterator it = The.begin(); it != The.end(); it++)
        Nexus<>::AddTaskVal(function, **it, std::ref(Args)...);
}

template<typename T>
template<typename N, typename F, typename ...A>
RIN xvector<N> PtrXVector<T*>::ForEach(F&& function, A&& ...Args)
{
    xvector<N> vret;
    for (typename xvector<E*>::iterator it = The.begin(); it != The.end(); it++)
        vret.push_back(function(**it, std::forward<A>(Args)...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
RIN std::unordered_map<K, V> PtrXVector<T*>::ForEach(F&& function, A&& ...Args)
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E*>::iterator it = The.begin(); it != The.end(); it++)
        rmap.insert(function(**it, std::forward<A>(Args)...));
    return rmap;
}
template<typename T>
template<typename N, typename F, typename ...A>
RIN xvector<N> PtrXVector<T*>::ForEach(F&& function, A&& ...Args) const
{
    xvector<N> vret;
    for (typename xvector<E*>::const_iterator it = The.begin(); it != The.end(); it++)
        vret.push_back(function(**it, std::forward<A>(Args)...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
RIN std::unordered_map<K, V> PtrXVector<T*>::ForEach(F&& function, A&& ...Args) const
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E*>::const_iterator it = The.begin(); it != The.end(); it++)
        rmap.insert(function(**it, std::forward<A>(Args)...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
RIN xvector<xp<N>> PtrXVector<T*>::ForEachThread(F&& function, A&& ...Args)
{
    if constexpr (std::is_same_v<N, std::remove_pointer_t<T>>)
    {
        SSET(MoNexus, MKP<Nexus<T>>());
        MoNexus.Disable();
        for (typename xvector<E*>::iterator it = The.begin(); it != The.end(); it++)
            MoNexus.AddTaskVal(function, **it, std::ref(Args)...);
        MoNexus.Enable();
        return MoNexus.GetAllPtrs();
    }
    else
    {
        Nexus<N> LoNexus;
        LoNexus.Disable();
        for (typename xvector<E*>::iterator it = The.begin(); it != The.end(); it++)
            LoNexus.AddTaskVal(function, **it, std::ref(Args)...);
        LoNexus.Enable();
        return LoNexus.GetAllPtrs();
    }
}

template<typename T>
template<typename N, typename F, typename ...A>
RIN xvector<xp<N>> PtrXVector<T*>::ForEachThread(F&& function, A&& ...Args) const
{
    Nexus<N> LoNexus;
    LoNexus.Disable();
    for (typename xvector<E*>::const_iterator it = The.begin(); it != The.end(); it++)
        LoNexus.AddTaskVal(function, **it, std::ref(Args)...);
    LoNexus.Enable();
    return LoNexus.GetAllPtrs();
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
RIN std::unordered_map<K, V> PtrXVector<T*>::ForEachThread(F&& function, A&& ...Args) const
{
    Nexus<std::pair<K, V>> LoNexus;
    LoNexus.Disable();
    for (typename xvector<E*>::iterator it = The.begin(); it != The.end(); it++)
        LoNexus.AddTaskVal(function, **it, std::ref(Args)...);
    LoNexus.Enable();
    return LoNexus.GetMoveAllIndices();
}

template<typename T>
template <typename N, typename F, typename ...A>
RIN void PtrXVector<T*>::StartTasks(F&& function, A&& ...Args)
{
    SSET(MoNexus, MKP<Nexus<T>>());
    MoNexus.Disable();
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
        MoNexus.AddTaskVal(function, **it, std::ref(Args)...);
    MoNexus.Enable();
}

template<typename T>
template<typename N>
RIN xvector<xp<N>> PtrXVector<T*>::GetTasks() const
{
    if (!The.MoNexusPtr)
        return std::vector<N>();
    return The.MoNexusPtr->GetAllPtrs();
}

template<typename T>
RIN bool PtrXVector<T*>::TasksCompleted() const
{
    if (!The.MoNexusPtr)
        return true;
    return The.MoNexusPtr->TasksCompleted();
}

// =============================================================================================================

template<typename T>
RIN T PtrXVector<T*>::GetSum(xint FnSkipIdx) const
{
    if (!Size())
        return 0;

    T LnModSize = 0;
    if (FnSkipIdx && Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 0;
    for (typename xvector<T*>::const_iterator it = The.begin() + LnModSize; it != The.end(); it++) {
        num += **it;
    }
    return num;
}

template<typename T>
RIN T PtrXVector<T*>::GetMul(xint FnSkipIdx) const
{
    if (!Size())
        return 0;

    if (Size() == 1)
        return (*this)[0];

    T LnModSize = 0;
    if (Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 1;
    for (typename xvector<T*>::const_iterator it = The.begin() + LnModSize; it != The.end(); it++) {
        num *= (**it);
    }
    return num;
}

template<typename T>
RIN T PtrXVector<T*>::GetAvg(xint FnSkipIdx) const
{
    return The.Sum(FnSkipIdx) / (The.Size() - FnSkipIdx);
}

// =============================================================================================================

template<typename T>
T PtrXVector<T*>::Join(const T& str) const
{
    T ret;
    for (typename PtrXVector<T*>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += **it + str;

    return ret.substr(0, ret.length() - str.size());
}

template<typename T>
T PtrXVector<T*>::Join(const char str) const
{
    T ret;
    for (typename PtrXVector<T*>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += **it + str;

    return ret.substr(0, ret.length() - 1);
}

template<typename T>
T PtrXVector<T*>::Join(const char* str) const
{
    T ret;
    for (typename PtrXVector<T*>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += **it + str;

    return ret.substr(0, ret.length() - strlen(str));
}

// =============================================================================================================

#ifndef UsingNVCC
template<typename T>
RIN bool PtrXVector<T*>::FullMatchOne(const re2::RE2& in_pattern) const
{
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (RE2::FullMatch(**iter, in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
RIN bool PtrXVector<T*>::FullMatchAll(const re2::RE2& in_pattern) const
{
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!RE2::FullMatch(**iter, in_pattern)) {
            return false;
        }
    }
    return true;
}


template<typename T>
RIN bool PtrXVector<T*>::MatchOne(const re2::RE2& in_pattern) const
{
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (RE2::PartialMatch(**iter, in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
RIN bool PtrXVector<T*>::MatchAll(const re2::RE2& in_pattern) const
{
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!RE2::PartialMatch(**iter, in_pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
RIN xvector<T*> PtrXVector<T*>::Take(const re2::RE2& in_pattern) const
{
    xvector<T*> RetVec;
    RetVec.reserve(The.size() + 1);
    for (xint i = 0; i < The.size(); i++) {
        if ((RE2::PartialMatch(*(*this)[i], in_pattern)))
            RetVec.push_back((*this)[i]);
    }
    return RetVec;
}

template<typename T>
RIN xvector<T*> PtrXVector<T*>::Remove(const re2::RE2& in_pattern) const
{
    xvector<T*> RetVec;
    RetVec.reserve(The.size() + 1);
    for (xint i = 0; i < The.size(); i++) {
        if (!(RE2::PartialMatch(*(*this)[i], in_pattern)))
            RetVec.push_back((*this)[i]);
    }
    return RetVec;
}

template<typename T>
RIN xvector<T> PtrXVector<T*>::SubAll(const re2::RE2& in_pattern, const std::string& replacement) const
{
    xvector<E> RetVec;
    RetVec.reserve(The.size() + 1);
    for (const T* Val : *this)
        RetVec << *Val;

    for (typename PtrXVector<T*>::iterator iter = RetVec.begin(); iter != RetVec.end(); iter++)
        RE2::GlobalReplace(&*iter, in_pattern, replacement.c_str());
    return RetVec;
}
#endif

// =============================================================================================================


template<typename T>
bool PtrXVector<T*>::FullMatchOne(const std::string& in_pattern) const 
{
#ifdef UsingNVCC
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (std::regex_match(**iter, rex)) {
            return true;
}
    }
    return false;
#else
    The.FullMatchOne(re2::RE2(in_pattern));
#endif
}

template<typename T>
bool PtrXVector<T*>::FullMatchOne(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.FullMatchOne(std::string(in_pattern));
#else
    return The.FullMatchOne(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool PtrXVector<T*>::FullMatchAll(const std::string& in_pattern) const 
{
#ifdef UsingNVCC
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!std::regex_match(**iter, rex)) {
            return false;
        }
    }
    return true;
#else
    The.FullMatchAll(re2::RE2(in_pattern));
#endif
}

template<typename T>
bool PtrXVector<T*>::FullMatchAll(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.FullMatchAll(std::string(in_pattern));
#else
    return The.FullMatchAll(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool PtrXVector<T*>::MatchOne(const std::string& in_pattern) const 
{
#ifdef UsingNVCC
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (std::regex_match(**iter, rex)) {
            return true;
        }
    }
    return false;
#else
    The.MatchOne(re2::RE2(in_pattern));
#endif
}

template<typename T>
bool PtrXVector<T*>::MatchOne(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.MatchOne(std::string(in_pattern));
#else
    return The.MatchOne(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool PtrXVector<T*>::MatchAll(const std::string& in_pattern) const {
#ifdef UsingNVCC
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!std::regex_match(**iter, rex)) {
            return false;
        }
    }
    return true;
#else
    The.MatchAll(re2::RE2(in_pattern));
#endif
}

template<typename T>
bool PtrXVector<T*>::MatchAll(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.MatchAll(std::string(in_pattern));
#else
    return The.MatchAll(re2::RE2(in_pattern));
#endif
}
// =============================================================================================================

template<typename T>
RIN xvector<T*> PtrXVector<T*>::Take(const std::string& in_pattern) const
{
#ifdef UsingNVCC
    xvector<T*> RetVec;
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (std::regex_match(**iter, rex)) {
            RetVec.push_back(*iter);
        }
    }
    return RetVec;
#else
    The.Take(re2::RE2(in_pattern));
#endif
}

template<typename T>
RIN xvector<T*> PtrXVector<T*>::Take(char const* in_pattern) const
{
#ifdef UsingNVCC
    return The.Take(std::string(in_pattern));
#else
    return The.Take(re2::RE2(in_pattern));
#endif
}

template<typename T>
RIN xvector<T*> PtrXVector<T*>::Remove(const std::string& in_pattern) const
{
#ifdef UsingNVCC
    xvector<T*> RetVec;
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!std::regex_match(**iter, rex)) {
            RetVec.push_back(*iter);
        }
    }
    return RetVec;
#else
    The.Remove(re2::RE2(in_pattern));
#endif
}


template<typename T>
RIN xvector<T*> PtrXVector<T*>::Remove(char const* in_pattern) const
{
#ifdef UsingNVCC
    return The.Remove(std::string(in_pattern));
#else
    return The.Remove(re2::RE2(in_pattern));
#endif
}
// =============================================================================================================

template<typename T>
RIN xvector<T> PtrXVector<T*>::SubAll(const std::string& Pattern, const std::string& Replacement) const
{
#ifdef UsingNVCC
    xvector<T> RetVec;
    std::regex rex(Pattern, RXM::ECMAScript);
    for (typename PtrXVector<xp<T>>::iterator iter = The.begin(); iter != The.end(); iter++)
        RetVec.push_back(std::regex_replace(**iter, rex, Replacement));
    return RetVec;
#else
    return The.SubAll(re2::RE2(Pattern.c_str()), Replacement);
#endif
}

template<typename T>
RIN xvector<T> PtrXVector<T*>::SubAll(char const* in_pattern, char const* replacement) const
{
    return The.SubAll(std::string(in_pattern), std::string(replacement));
}

// =============================================================================================================

template<typename T>
xvector<T*> PtrXVector<T*>::operator()(long double x, long double y, long double z, const char removal_method) const {

    xint m_size = The.size();
    xvector<T*> n_arr;
    n_arr.reserve(m_size + 4);

    double n_arr_size = static_cast<double>(m_size) - 1;

    if (z >= 0) {

        if (x < 0) { x += n_arr_size; }

        if (!y) { y = n_arr_size; }
        else if (y < 0) { y += n_arr_size; }
        ++y;

        if (x > y) { return n_arr; }

        typename xvector<T*>::const_iterator iter = The.begin();
        typename xvector<T*>::const_iterator stop = The.begin() + static_cast<xint>(y);

        if (z == 0) { // forward direction with no skipping
            for (iter += static_cast<xint>(x); iter != stop; ++iter)
                n_arr.push_back(*iter);
        }
        else if (removal_method == 's') { // forward direction with skipping
            double iter_insert = 0;
            --z;
            for (iter += static_cast<xint>(x); iter != stop; ++iter) {
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
            for (iter += static_cast<xint>(x); iter != stop; ++iter) {
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

        typename xvector<T*>::const_reverse_iterator iter = The.rend() - static_cast<xint>(x) - 1;
        typename xvector<T*>::const_reverse_iterator stop = The.rend() - static_cast<xint>(y);

        xint iter_insert = 0;

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
                    iter_insert = static_cast<xint>(z);
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = static_cast<xint>(z);
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



