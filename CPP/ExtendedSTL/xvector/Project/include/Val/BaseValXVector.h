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
#include <vector>
#include <type_traits>

TTT class ValXVector;
TTT std::ostream& operator<<(std::ostream& out, const ValXVector<T>& FVector);

template<typename T>
class ValXVector : public BaseXVector<T>
{
protected:
    xp<Nexus<T>> MoNexusPtr = nullptr;
public:
    using BaseXVector<T>::BaseXVector;
    using BaseXVector<T>::operator=;
    using E = typename std::remove_const<T>::type; // E for Erratic

    RIN void Print() const { std::cout << The << std::endl; }

    RIN T& At(const xint Idx);
    RIN const T& At(const xint Idx) const;

    template<typename P = T> 
    RIN bool Has(const P* Item) const;
    RIN bool Has(const T& Item) const;
    RIN bool Has(char const* Item) const;
    template<typename F, typename ...A>
    RIN bool HasTruth(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    RIN const T& GetTruth(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    RIN T& GetTruth(F&& Function, A&& ...Args);
    template<typename F, typename ...A>
    RIN xvector<T> GetTruths(F&& Function, A&& ...Args) const;

    RIN bool Lacks(const T& Item) const;
    RIN bool Lacks(char const* Item) const;
    template<typename F, typename ...A>
    RIN bool LacksTruth(F&& Function, A&& ...Args) const;

    template<typename L = xvector<T>>
    RIN xvector<T> GetCommonItems(L& Item);
    template<typename L = xvector<T>>
    RIN xvector<T> GetCommonItems(L&& Item);

    RIN void operator<<(const T&  Item);
    RIN void operator<<(      T&& Item);

    RIN void AddCharStrings(int strC, char** strV);

    T& First(xint Idx = 0);
    const T& First(xint Idx = 0) const;

    T& Last(xint Idx = 0);
    const T& Last(xint Idx = 0) const;

    RIN std::pair<T, T> GetPair() const;

    template<typename C>
    RIN xvector<C> Convert() const;

    template<typename C, typename F>
    RIN xvector<C> Convert(F FfFunction) const;

    template<typename N = unsigned int>
    RIN xvector<xvector<T>> Split(N count = 1) const;

    RIN void operator+=(const xvector<T>& other);
    RIN xvector<T> operator+(const xvector<T>& other) const;

    constexpr xint Size() const { return The.size(); }

    RIN void Organize();
    RIN void RemoveDups();

    template<typename F>
    RIN xvector<T>& Sort(F func);
    RIN xvector<T>& Sort();
    RIN xvector<T>& ReverseSort();
    RIN xvector<T>& ReverseIt();

    RIN xvector<T*> GetPtrs();

    template<typename F, typename... A>
    RIN void Proc(F&& FfFunction, A&& ...Args);
    template<typename F, typename... A>
    RIN void ProcThread(F&& FfFunction, A&& ...Args);

    // single threaded non-const
    template<typename N = E, typename F, typename ...A>             RIN xvector<N> ForEach(F&& FfFunction, A&& ...Args);
    template<typename K, typename V, typename F, typename ...A>     RIN std::unordered_map<K, V> ForEach(F&& FfFunction, A&& ...Args);
    // single threaded const
    template<typename N = E, typename F, typename ...A>             RIN xvector<N> ForEach(F&& FfFunction, A&& ...Args) const;
    template<typename K, typename V, typename F, typename ...A>     RIN std::unordered_map<K, V> ForEach(F&& FfFunction, A&& ...Args) const;
    // multi-threaded const & non-const

    template<typename N = E, typename F>                            RIN xvector<N> ForEachThread(F&& FfFunction);
    template<typename N = E, typename F, typename A, typename ...B> RIN xvector<N> ForEachThread(F&& FfFunction, A&& FoFirst, B&& ...FoRest);
    //template<typename K, typename V, typename F, typename ...A>     RIN std::unordered_map<K, V> ForEachThread(F&& FfFunction, A&& ...Args) const;

    template<typename N = E, typename F, typename... A>             RIN void StartTasks(F&& FfFunction, A&& ...Args);

    template<typename N = E>                                        RIN xvector<xp<T>> GetTasks() const;

                                                                    RIN bool TasksCompleted() const;

    // =================================== DESIGNED FOR NUMERIC BASED VECTORS ===================================

    RIN T GetSum(xint FnSetBackIDX = 0) const;
    RIN T GetMul(xint FnSetBackIDX = 0) const;
    RIN T GetAvg(xint FnSetBackIDX = 0) const;

    // =================================== DESIGNED FOR STRING  BASED VECTORS ===================================

    RIN T Join(const T& str = "") const;
    RIN T Join(const char str) const;
    RIN T Join(const char* str) const;

#ifndef UsingNVCC
    RIN bool FullMatchOne(const re2::RE2& in_pattern) const;
    RIN bool FullMatchAll(const re2::RE2& in_pattern) const;
    RIN bool MatchOne(const re2::RE2& in_pattern) const;
    RIN bool MatchAll(const re2::RE2& in_pattern) const;
    RIN xvector<T> Take(const re2::RE2& in_pattern) const;
    RIN xvector<T> Remove(const re2::RE2& in_pattern) const;
    RIN xvector<T> SubAll(const re2::RE2& in_pattern, const std::string& replacement) const;
#endif // !UsingNVCC

    RIN bool FullMatchOne(const std::string& in_pattern) const;
    RIN bool FullMatchOne(char const* in_pattern) const;

    RIN bool FullMatchAll(const std::string& in_pattern) const;
    RIN bool FullMatchAll(char const* in_pattern) const;

    RIN bool MatchOne(const std::string& in_pattern) const;
    RIN bool MatchOne(char const* in_pattern) const;

    RIN bool MatchAll(const std::string& in_pattern) const;
    RIN bool MatchAll(char const* in_pattern) const;

    RIN xvector<T> TakePattern(const std::string& in_pattern) const;
    RIN xvector<T> RemovePattern(const std::string& in_pattern) const;

    RIN xvector<T> SubAll(const std::string& in_pattern, const std::string& replacement) const;
    RIN xvector<T> SubAll(char const* in_pattern, char const* replacement) const;
    
    // double was chose to hold long signed and unsigned values

    RIN xvector<T> operator()(const long long int x) const;
    RIN xvector<T> operator()(
        const long long int x,
        const long long int y,
        const long long int z = 0,
        const char removal_method = 's') const;
    // s = slice perserves values you land on 
    // d = dice  removes values you land on
    // s/d only makes a difference if you modify the 'z' value

    // =================================== DESIGNED FOR STRING BASED VECTORS ===================================
};
// =============================================================================================================

template<typename T>
RIN T& ValXVector<T>::At(const xint Idx)
{
    if (Idx >= Size())
        throw "Index Out Of Range";
    return The[Idx];
}

template<typename T>
RIN const T& ValXVector<T>::At(const xint Idx) const
{
    if (Idx >= Size())
        throw "Index Out Of Range";
    return The[Idx];
}

template<typename T>
template<typename P>
RIN bool ValXVector<T>::Has(const P* Item) const
{
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (**it == *Item)
            return true;
    }
    return false;
}

template<typename T>
RIN bool ValXVector<T>::Has(const T& Item) const{
    return (bool(std::find(The.begin(), The.end(), Item) != The.end()));
}

template<typename T>
RIN bool ValXVector<T>::Has(char const* Item) const {
    return (bool(std::find(The.begin(), The.end(), Item) != The.end()));
}

template<typename T>
template<typename F, typename ...A>
RIN bool ValXVector<T>::HasTruth(F&& Function, A&& ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function(*it, std::forward<A>(Args)...))
            return true;
    }
    return false;
}

template<typename T>
template<typename F, typename ...A>
RIN const T& ValXVector<T>::GetTruth(F&& Function, A && ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function(*it, std::forward<A>(Args)...))
            return (*it);
    }
    throw "Truth Not Found";
}

template<typename T>
template<typename F, typename ...A>
RIN T& ValXVector<T>::GetTruth(F&& Function, A && ...Args)
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function(*it, std::forward<A>(Args)...))
            return (*it);
    }
    throw "Truth Not Found";
}

template<typename T>
template<typename F, typename ...A>
RIN xvector<T> ValXVector<T>::GetTruths(F&& Function, A && ...Args) const
{
    xvector<T> RetVec;
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function(*it, std::forward<A>(Args)...))
            RetVec << (*it);
    }
    return RetVec;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
RIN bool ValXVector<T>::Lacks(const T& Item) const {
    return !(bool(std::find(The.begin(), The.end(), Item) != The.end()));
}

template<typename T>
RIN bool ValXVector<T>::Lacks(char const* Item) const {
    return !(bool(std::find(The.begin(), The.end(), Item) != The.end()));
}

template<typename T>
template<typename F, typename ...A>
RIN bool ValXVector<T>::LacksTruth(F&& Function, A && ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((*it), std::forward<A>(Args)...))
            return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------


template<typename T>
template<typename L>
xvector<T> ValXVector<T>::GetCommonItems(L& Item) 
{
    std::sort(The.begin(), The.end());
    std::sort(Item.begin(), Item.end());

    xvector<T> vret(The.size() + Item.size());
    set_intersection(The.begin(), The.end(), Item.begin(), Item.end(), vret.begin());
    return vret;
}

template<typename T>
template<typename L>
xvector<T> ValXVector<T>::GetCommonItems(L&& Item) {
    return The.GetCommonItems(Item);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
RIN void ValXVector<T>::operator<<(const T& Item)
{
    The.emplace_back(Item);
}

template<typename T>
RIN void ValXVector<T>::operator<<(T&& Item){
    The.emplace_back(std::forward<T>(Item));
}

template<typename T>
RIN void ValXVector<T>::AddCharStrings(int strC, char** strV)
{
    for (int i = 0; i < strC; i++)
        The.push_back(T(strV[i]));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
RIN T& ValXVector<T>::First(xint Idx)
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Bounds";
    return The.operator[](Idx);
}

template<typename T>
RIN const T& ValXVector<T>::First(xint Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Bounds";
    return The.operator[](Idx);
}

template<typename T>
RIN T& ValXVector<T>::Last(xint Idx)
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Bounds";
    return The.operator[](The.size() - Idx - 1);
}


template<typename T>
RIN const T& ValXVector<T>::Last(xint Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Index Out Of Bounds";
    return The.operator[](The.size() - Idx - 1);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
RIN std::pair<T, T> ValXVector<T>::GetPair() const
{
    return std::pair<E, E>(The.at(0), The.at(1));
}

template<typename T>
template<typename C>
RIN xvector<C> ValXVector<T>::Convert() const
{
    xvector<C> ret;
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++)
        ret << C(*it);
    return ret;
}


template<typename T>
template<typename C, typename F>
RIN xvector<C> ValXVector<T>::Convert(F FfFunction) const
{
    xvector<C> ret;
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++)
        ret << FfFunction(*it);
    return ret;
}

template<typename T>
template<typename N>
RIN xvector<xvector<T>> ValXVector<T>::Split(N count) const
{

    xvector<xvector<T>> RetVec;
    if (count < 2) {
        if (count == 1 && The.size() == 1) {
            RetVec[0].reserve(The.size());
            for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++) {
                RetVec[0].push_back(*it);
            }
        }
        else
            return RetVec;
    }

    RetVec.reserve(static_cast<xint>(count) + 1);
    if (!The.size())
        return RetVec;

    N reset = count;
    count = 0;
    const N new_size = static_cast<N>(The.size()) / reset;
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (count == 0) {
            count = reset;
            RetVec.push_back(xvector<T>({ *it })); // create new xvec and add first el
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
void ValXVector<T>::operator+=(const xvector<T>& other)
{
    The.insert(The.end(), other.begin(), other.end());
}

template<typename T>
xvector<T> ValXVector<T>::operator+(const xvector<T>& other) const 
{
    xvector<T> vret;
    vret.reserve(The.size());
    vret.insert(vret.end(),  The.begin(), The.end());
    vret += other;
    return vret;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
RIN void ValXVector<T>::Organize()
{
    std::multiset<T> set_arr;
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++)
        set_arr.insert(*it);

    The.clear();
    The.reserve(set_arr.size());

    for (typename std::multiset<T>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        The.push_back(*it);
}

template<typename T>
RIN void ValXVector<T>::RemoveDups()
{
    std::set<T> set_arr;
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++)
        set_arr.insert(*it);

    The.clear();
    The.reserve(set_arr.size());

    for (typename std::set<T>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        The.push_back(*it);
}

// -----------------------------------------------------------------------------------------------

template<typename T>
template<typename F>
RIN xvector<T>& ValXVector<T>::Sort(F func)
{
    std::sort(The.begin(), The.end(), func);
    return *reinterpret_cast<xvector<T>*>(this);
}

template<typename T>
RIN xvector<T>& ValXVector<T>::Sort()
{
    std::sort(The.begin(), The.end());
    return *reinterpret_cast<xvector<T>*>(this);
}

template<typename T>
RIN xvector<T>& ValXVector<T>::ReverseSort()
{
    std::sort(The.begin(), The.end(), std::greater<T>());
    return *reinterpret_cast<xvector<T>*>(this);
}

template<typename T>
RIN xvector<T>& ValXVector<T>::ReverseIt()
{
    std::reverse(The.begin(), The.end());
    return *reinterpret_cast<xvector<T>*>(this);
}

template<typename T>
RIN xvector<T*> ValXVector<T>::GetPtrs()
{
    xvector<T*> RetVec;
    for (T& Item : *this)
        RetVec << &Item;

    return RetVec;
}

template<typename T>
template<typename F, typename... A>
RIN void ValXVector<T>::Proc(F&& FfFunction, A&& ...Args)
{
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++) {
        if (FfFunction(*it, std::forward<A>(Args)...))
            break;
    }
}

template<typename T>
template<typename F, typename... A>
RIN void ValXVector<T>::ProcThread(F&& FfFunction, A&& ...Args)
{
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
        Nexus<>::AddTaskVal(FfFunction, *it, std::ref(Args)...);
}

template<typename T>
template<typename N, typename F, typename ...A>
RIN xvector<N> ValXVector<T>::ForEach(F&& FfFunction, A&& ...Args) const
{
    xvector<N> vret;
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++)
        vret.push_back(FfFunction(*it, std::forward<A>(Args)...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
RIN std::unordered_map<K, V> ValXVector<T>::ForEach(F&& FfFunction, A&& ...Args)
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
        rmap.insert(FfFunction(*it, std::forward<A>(Args)...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
RIN xvector<N> ValXVector<T>::ForEach(F&& FfFunction, A&& ...Args)
{
    xvector<N> vret;
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
        vret.push_back(FfFunction(*it, std::forward<A>(Args)...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
RIN std::unordered_map<K, V> ValXVector<T>::ForEach(F&& FfFunction, A&& ...Args) const
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++)
        rmap.insert(FfFunction(*it, std::forward<A>(Args)...));
    return rmap;
}

template<typename T>
template<typename N, typename F>
RIN xvector<N> ValXVector<T>::ForEachThread(F&& FfFunction)
{
#ifdef UsingMSVC
    //if constexpr (sizeof...(Args) == 0)
        return The.ForEachThreadUnseq<N>(FfFunction);
#else
    auto Err = std::string("Can't use: " __CLASS__);
    cout << Err << endl;
    throw Err;
#endif
}

template<typename T>
template<typename N, typename F, typename A, typename ...B>
RIN xvector<N> ValXVector<T>::ForEachThread(F&& FfFunction, A&& FoFirst, B&& ...FoRest)
{
    if constexpr (BxSameType(N, T))
    {
        SSET(MoNexus, MKP<Nexus<T>>());
        MoNexus.Disable();
        for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
            MoNexus.AddTaskVal(FfFunction, *it, std::ref(FoFirst), std::ref(FoRest)...);
        MoNexus.Enable();
        return MoNexus.GetAllPtrs();
    }
    else
    {
        Nexus<N> LoNexus;
        LoNexus.Disable();
        for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
            LoNexus.AddTaskVal(FfFunction, *it, std::ref(FoFirst), std::ref(FoRest)...);
        LoNexus.Enable();
        return LoNexus.GetAllPtrs();
    }
}

//template<typename T>
//template<typename K, typename V, typename F, typename ...A>
//RIN std::unordered_map<K, V> ValXVector<T>::ForEachThread(F&& FfFunction, A&& ...Args) const
//{
//    auto MapPool = Nexus<std::pair<K, V>>();
//    MapPool.Disable();
//    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++)
//        MapPool.AddTaskVal(FfFunction, *it, std::ref(Args)...);
//    MapPool.Enable();
//    return MapPool.GetMoveAllIndices();
//}

template<typename T>
template <typename N, typename F, typename ...A>
RIN void ValXVector<T>::StartTasks(F&& FfFunction, A&& ...Args)
{
    SSET(MoNexus, MKP<Nexus<T>>());
    MoNexus.Disable();
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
        MoNexus.AddTaskVal(FfFunction, *it, std::ref(Args)...);
    MoNexus.Enable();
}

template<typename T>
template<typename N>
RIN xvector<xp<T>> ValXVector<T>::GetTasks() const
{
    if (!The.MoNexusPtr)
        return std::vector<xp<N>>();
    return The.MoNexus->GetAllPtrs();
}

template<typename T>
RIN bool ValXVector<T>::TasksCompleted() const
{
    if (!The.MoNexusPtr)
        return true;
    return The.MoNexus->TasksCompleted();
}

// =============================================================================================================


template<typename T>
RIN T ValXVector<T>::GetSum(xint FnSetBackIDX) const
{
    if (!Size())
        return 0;

    xint SetBackIDX = 0;
    if (FnSetBackIDX)
        SetBackIDX = (Size() > FnSetBackIDX) ? FnSetBackIDX : Size();
    else
        SetBackIDX = Size();

    T num = 0;
    for (typename ValXVector<T>::const_iterator it = The.end() - SetBackIDX; it != The.end(); it++) {
        num += *it;
    }
    return num;
}

template<typename T>
RIN T ValXVector<T>::GetMul(xint FnSetBackIDX) const
{
    if (!Size())
        return 0;

    if (Size() == 1)
        return The[0];

    const xint SetBackIDX = (Size() > FnSetBackIDX) ? FnSetBackIDX : Size();

    T num = 1;
    for (typename ValXVector<T>::const_iterator it = The.end() - SetBackIDX; it != The.end(); it++) {
        num *= (*it);
    }
    return num;
}

template<typename T>
RIN T ValXVector<T>::GetAvg(xint FnSetBackIDX) const
{
    if (!Size())
        return 0;

    xint SetBackIDX = 0;
    if (FnSetBackIDX)
        SetBackIDX = (Size() > FnSetBackIDX) ? FnSetBackIDX : Size();
    else
        SetBackIDX = Size();

    return The.GetSum(SetBackIDX) / SetBackIDX;
}

// =============================================================================================================

template<typename T>
RIN T ValXVector<T>::Join(const T& str) const
{
    T ret;
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - str.size());
}

template<typename T>
T ValXVector<T>::Join(const char str) const
{
    T ret;
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - 1);
}

template<typename T>
T ValXVector<T>::Join(const char* str) const
{
    T ret;
    for (typename ValXVector<T>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - strlen(str));
}
// =============================================================================================================

#ifndef UsingNVCC
template<typename T>
RIN bool ValXVector<T>::FullMatchOne(const re2::RE2& in_pattern) const
{
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (RE2::FullMatch(*iter, in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
RIN bool ValXVector<T>::FullMatchAll(const re2::RE2& in_pattern) const
{
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!RE2::FullMatch(*iter, in_pattern)) {
            return false;
        }
    }
    return true;
}


template<typename T>
RIN bool ValXVector<T>::MatchOne(const re2::RE2& in_pattern) const
{
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (RE2::PartialMatch(*iter, in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
RIN bool ValXVector<T>::MatchAll(const re2::RE2& in_pattern) const
{
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!RE2::PartialMatch(*iter, in_pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
RIN xvector<T> ValXVector<T>::Take(const re2::RE2& in_pattern) const
{
    xvector<T> RetVec;
    RetVec.reserve(The.size() + 1);
    for (xint i = 0; i < The.size(); i++) {
        if ((RE2::PartialMatch(The[i], in_pattern)))
            RetVec.push_back(The[i]);
    }
    return RetVec;
}

template<typename T>
RIN xvector<T> ValXVector<T>::Remove(const re2::RE2& in_pattern) const
{
    xvector<T> RetVec;
    RetVec.reserve(The.size() + 1);
    for (xint i = 0; i < The.size(); i++) {
        if (!(RE2::PartialMatch(The[i], in_pattern)))
            RetVec.push_back(The[i]);
    }
    return RetVec;
}

template<typename T>
RIN xvector<T> ValXVector<T>::SubAll(const re2::RE2& in_pattern, const std::string& replacement) const
{
    xvector<E> RetVec;
    RetVec.reserve(The.size() + 1);
    for (const T& Val : *this)
        RetVec << Val;

    for (typename ValXVector<T>::iterator iter = RetVec.begin(); iter != RetVec.end(); iter++)
        RE2::GlobalReplace(&*iter, in_pattern, replacement.c_str());
    return RetVec;
}
#endif

// =============================================================================================================


template<typename T>
bool ValXVector<T>::FullMatchOne(const std::string& in_pattern) const 
{
#ifdef UsingNVCC
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (std::regex_match(*iter, rex)) {
            return true;
}
    }
    return false;
#else
    The.FullMatchOne(re2::RE2(in_pattern));
#endif
}

template<typename T>
bool ValXVector<T>::FullMatchOne(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.FullMatchOne(std::string(in_pattern));
#else
    return The.FullMatchOne(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool ValXVector<T>::FullMatchAll(const std::string& in_pattern) const 
{
#ifdef UsingNVCC
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!std::regex_match(*iter, rex)) {
            return false;
        }
    }
    return true;
#else
    The.FullMatchAll(re2::RE2(in_pattern));
#endif
}

template<typename T>
bool ValXVector<T>::FullMatchAll(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.FullMatchAll(std::string(in_pattern));
#else
    return The.FullMatchAll(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool ValXVector<T>::MatchOne(const std::string& in_pattern) const 
{
#ifdef UsingNVCC
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (std::regex_match(*iter, rex)) {
            return true;
        }
    }
    return false;
#else
    return The.MatchOne(re2::RE2(in_pattern));
#endif
}

template<typename T>
bool ValXVector<T>::MatchOne(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.MatchOne(std::string(in_pattern));
#else
    return The.MatchOne(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool ValXVector<T>::MatchAll(const std::string& in_pattern) const {
#ifdef UsingNVCC
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!std::regex_match(*iter, rex)) {
            return false;
        }
    }
    return true;
#else
    The.MatchAll(re2::RE2(in_pattern));
#endif
}

template<typename T>
bool ValXVector<T>::MatchAll(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.MatchAll(std::string(in_pattern));
#else
    return The.MatchAll(re2::RE2(in_pattern));
#endif
}
// =============================================================================================================

template<typename T>
RIN xvector<T> ValXVector<T>::TakePattern(const std::string& in_pattern) const
{
#ifdef UsingNVCC
    xvector<T> RetVec;
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (std::regex_match(*iter, rex)) {
            RetVec.push_back(*iter);
        }
    }
    return RetVec;
#else
    return The.Take(re2::RE2(in_pattern));
#endif
}

template<typename T>
RIN xvector<T> ValXVector<T>::RemovePattern(const std::string& in_pattern) const
{
#ifdef UsingNVCC
    xvector<T> RetVec;
    std::regex rex(in_pattern, RXM::ECMAScript);
    for (typename xvector<T>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!std::regex_match(*iter, rex)) {
            RetVec.push_back(*iter);
        }
    }
    return RetVec;
#else
    The.Remove(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
RIN xvector<T> ValXVector<T>::SubAll(const std::string& Pattern, const std::string& Replacement) const
{
#ifdef UsingNVCC
    xvector<T> RetVec = The;
    std::regex rex(Pattern, RXM::ECMAScript);
    for (typename ValXVector<T>::iterator iter = RetVec.begin(); iter != RetVec.end(); iter++)
        RetVec.push_back(std::regex_replace(*iter, rex, Replacement));
    return RetVec;
#else
    return The.SubAll(re2::RE2(Pattern.c_str()), Replacement);
#endif
}

template<typename T>
RIN xvector<T> ValXVector<T>::SubAll(char const* in_pattern, char const* replacement) const
{
    return The.SubAll(std::string(in_pattern), std::string(replacement));
}

// =============================================================================================================


template<typename T>
xvector<T> ValXVector<T>::operator()(const long long int x) const
{
    if (x == 0)
        return The;

    if (x > 0 && x >= The.size())
        return xvector<T>{};

    if (x < 0 && std::abs(x) > The.size())
        return The;

    else if (x > 0)
    {
        // return The.substr(x, size() - x);
        xvector<T> ret;
        typename ValXVector<T>::const_iterator it = The.begin();
        it += x;
        for (; it != The.end(); it++)
            ret.push_back(*it);
        return ret;
    }
    else
    {
        // return The.substr(size() + x, size() - (size() + x));
        xvector<T> ret;
        typename ValXVector<T>::const_iterator it = The.begin();
        it += The.Size() + x;
        for (; it != The.end(); it++)
            ret.push_back(*it);
        return ret;
    }
}

template<typename T>
xvector<T> ValXVector<T>::operator()(
    const long long int x,
    const long long int y,
    const long long int z,
    const char removal_method) const
{
    const auto m_size = static_cast<long long int>(The.size());
    if (m_size <= 1)
        return The;

    xvector<T> n_arr;
    n_arr.reserve(m_size + 4);

    if (z >= 0)
    {
        const auto tx = (x >= 0) ? x : m_size + x + 1;
        const auto ty = (y >= 0) ? y : m_size + y;

        typename xvector<T>::const_iterator iter = The.begin() + tx;
        typename xvector<T>::const_iterator stop = The.begin() + ty;

        if (z == 0) { // forward direction with no skipping
            for (; iter != stop; ++iter)
                n_arr.push_back(*iter);
        }
        else if (removal_method == 's') { // forward direction with skipping
            double iter_insert = 0;
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    n_arr.push_back(*iter);
                    iter_insert = z - 1;
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            double iter_insert = 0;
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = z - 1;
                }
                else {
                    n_arr.push_back(*iter);
                    --iter_insert;
                }
            }
        }
    }
    else { // reverse direction
        const auto tx = (x >= 0) ? m_size - x - 1 : std::abs(x) - 1;
        const auto ty = (y >= 0) ? m_size - y - 1 : std::abs(y) - 1;

        typename xvector<T>::const_reverse_iterator iter = The.rbegin() + tx;
        typename xvector<T>::const_reverse_iterator stop = The.rbegin() + ty;

        double iter_insert = 0;

        if (z + 1 == 0) {
            for (; iter != stop; ++iter) {
                n_arr.push_back(*iter);
            }
        }
        else if (removal_method == 's') {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    n_arr.push_back(*iter);
                    iter_insert = z + 1;
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = z + 1;
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

TTT std::ostream& operator<<(std::ostream& out, const ValXVector<T>& FVector)
{
    out << '[';
    const auto LnSize = FVector.Size();
    for (xint i = 0; i < LnSize;)
    {
        out << FVector[i];
        if (++i < LnSize)
            out << ',';
    }
    out << ']';
    return out;
}
