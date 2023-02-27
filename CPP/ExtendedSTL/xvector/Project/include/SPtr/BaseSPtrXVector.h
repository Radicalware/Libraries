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
    using E = typename std::remove_const<xp<T>>::type; // E for Erratic

    inline const T& operator[](const size_t Idx) const;
    inline T& operator[](const size_t Idx);
    inline T& At(const size_t Idx);
    inline const T& At(const size_t Idx) const;
    inline const xp<T> AtPtr(const size_t Idx) const;
    inline xp<T> AtPtr(const size_t Idx);
    template<typename C = T>
    inline xvector<C> GetValues() const;
    inline xvector<T> GetValues() const;

    template<typename P = T> 
    inline bool Has(const P* item) const;
    inline bool Has(const T& item) const;
    inline bool Has(char const* item) const;

    template<typename F, typename ...A>
    inline bool HasTruth(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    inline const T& GetTruth(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    inline const xp<T> GetTruthPtr(F&& Function, A&& ...Args) const;
    template<typename F, typename ...A>
    inline T& GetTruth(F&& Function, A&& ...Args);
    template<typename F, typename ...A>
    inline xvector<xp<T>> GetTruths(F&& Function, A&& ...Args) const;

    inline bool Lacks(const T& item) const;
    inline bool Lacks(char const* item) const;
    template<typename F, typename ...A>
    inline bool LacksTruth(F&& Function, A&& ...Args) const;

    template<typename L = xvector<xp<T>>>
    inline xvector<xp<T>> GetCommonItems(L& item);
    template<typename L = xvector<xp<T>>>
    inline xvector<xp<T>> GetCommonItems(L&& item);

    template<typename ...R>
    inline void Emplace(R&& ... Args);
    inline void operator<<(const xp<T>&  item);
    inline void operator<<(      xp<T>&& item);

    T& First(size_t Idx = 0);
    const T& First(size_t Idx = 0) const;

    T& Last(size_t Idx = 0);
    const T& Last(size_t Idx = 0) const;

    template<typename C>
    inline xvector<C> Convert() const;

    template<typename C, typename F>
    inline xvector<C> Convert(F function) const;

    inline void operator+=(const xvector<xp<T>>& other);
    inline xvector<xp<T>> operator+(const xvector<xp<T>>& other) const;

    size_t Size() const;

    inline void Organize();
    inline void RemoveDups();

    template<typename F>
    inline xvector<xp<T>>& Sort(F&& func);
    inline xvector<xp<T>>& Sort();
    template<typename F>
    inline xvector<xp<T>>& ReverseSort(F&& func);
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

#ifndef UsingNVCC
    inline bool FullMatchOne(const re2::RE2& in_pattern) const;
    inline bool FullMatchAll(const re2::RE2& in_pattern) const;
    inline bool MatchOne(const re2::RE2& in_pattern) const;
    inline bool MatchAll(const re2::RE2& in_pattern) const;
    inline xvector<xp<T>> Take(const re2::RE2& in_pattern) const;
    inline xvector<xp<T>> Remove(const re2::RE2& in_pattern) const;
    inline xvector<T> SubAll(const re2::RE2& in_pattern, const std::string& replacement) const;
#endif // !UsingNVCC

    inline bool FullMatchOne(const std::string& Pattern) const;
    inline bool FullMatchOne(char const* Pattern) const;

    inline bool FullMatchAll(const std::string& Pattern) const;
    inline bool FullMatchAll(char const* Pattern) const;

    inline bool MatchOne(const std::string& Pattern) const;
    inline bool MatchOne(char const* Pattern) const;

    inline bool MatchAll(const std::string& Pattern) const;
    inline bool MatchAll(char const* Pattern) const;

    inline xvector<xp<T>> Take(const std::string& Pattern) const;
    inline xvector<xp<T>> Take(char const* Pattern) const;

    inline xvector<xp<T>> Remove(const std::string& Pattern) const;
    inline xvector<xp<T>> Remove(char const* Pattern) const;

    inline xvector<T> SubAll(const std::string& Pattern, const std::string& Replacement) const;
    inline xvector<T> SubAll(char const* Pattern, char const* Replacement)  const;

    inline void InSubAll(const xp<T>& Pattern, const xp<T>& Replacement);
    inline void InSubOne(const xp<T>& Pattern, const xp<T>& Replacement);

    // double was chose to hold long signed and unsigned values
    inline xvector<xp<T>> operator()(long double x = 0, long double y = 0, long double z = 0, const char removal_method = 's') const;
    // s = slice perserves values you land on 
    // d = dice  removes values you land on
    // s/d only makes a difference if you modify the 'z' value

    // =================================== DESIGNED FOR STRING BASED VECTORS ===================================
};
// =============================================================================================================

template<typename T>
inline const T& SPtrXVector<xp<T>>::operator[](const size_t Idx) const
{
    return The.AtPtr(Idx).Get();
}

template<typename T>
inline T& SPtrXVector<xp<T>>::operator[](const size_t Idx)
{
    return The.AtPtr(Idx).Get();
}

template<typename T>
inline T& SPtrXVector<xp<T>>::At(const size_t Idx)
{
    return The[Idx];
}

template<typename T>
inline const T& SPtrXVector<xp<T>>::At(const size_t Idx) const
{
    return The[Idx];
}

template<typename T>
inline const xp<T> SPtrXVector<xp<T>>::AtPtr(const size_t Idx) const
{
    if (Idx >= The.Size())
        throw "Index Out Of Bounds";
    return BaseXVector<xp<T>>::operator[](Idx);
}

template<typename T>
inline xp<T> SPtrXVector<xp<T>>::AtPtr(const size_t Idx)
{
    if (Idx >= The.Size())
        throw "Index Out Of Bounds";
    return BaseXVector<xp<T>>::operator[](Idx);
}

template<typename T>
template<typename C>
inline xvector<C> SPtrXVector<xp<T>>::GetValues() const
{
    xvector<C> Vec;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        Vec << static_cast<C>(**it);
    return Vec;
}

template<typename T>
inline xvector<T> SPtrXVector<xp<T>>::GetValues() const
{
    xvector<T> Vec;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        Vec << **it;
    return Vec;
}

template<typename T>
template<typename P>
inline bool SPtrXVector<xp<T>>::Has(const P* item) const
{
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (**it == *item)
            return true;
    }
    return false;
}

template<typename T>
bool SPtrXVector<xp<T>>::Has(const T& item) const {
    return (bool(std::find(The.begin(), The.end(), item) != The.end()));
}

template<typename T>
bool SPtrXVector<xp<T>>::Has(char const* item) const {
    return (bool(std::find(The.begin(), The.end(), item) != The.end()));
}

template<typename T>
template<typename F, typename ...A>
bool SPtrXVector<xp<T>>::HasTruth(F&& Function, A&& ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((*it).Get(), std::forward<A>(Args)...))
            return true;
    }
    return false;
}

template<typename T>
template<typename F, typename ...A>
inline const T& SPtrXVector<xp<T>>::GetTruth(F&& Function, A && ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((*it).Get(), std::forward<A>(Args)...))
            return (*it).Get();
    }
    throw "Truth Not Found";
}

template<typename T>
template<typename F, typename ...A>
inline const xp<T> SPtrXVector<xp<T>>::GetTruthPtr(F&& Function, A && ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((*it).Get(), std::forward<A>(Args)...))
            return (*it);
    }
    throw "Truth Not Found";
}

template<typename T>
template<typename F, typename ...A>
inline T& SPtrXVector<xp<T>>::GetTruth(F&& Function, A && ...Args)
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((*it).Get(), std::forward<A>(Args)...))
            return (*it).Get();
    }
    throw "Truth Not Found";
}
template<typename T>
template<typename F, typename ...A>
inline xvector<xp<T>> SPtrXVector<xp<T>>::GetTruths(F&& Function, A && ...Args) const
{
    xvector<T> RetVec;
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((*it).Get(), std::forward<A>(Args)...))
            RetVec >> *it;
    }
    throw "Truth Not Found";
}

// ------------------------------------------------------------------------------------------------

template<typename T>
bool SPtrXVector<xp<T>>::Lacks(const T& item) const {
    return !(bool(std::find(The.begin(), The.end(), item) != The.end()));
}

template<typename T>
bool SPtrXVector<xp<T>>::Lacks(char const* item) const {
    return !(bool(std::find(The.begin(), The.end(), item) != The.end()));
}

template<typename T>
template<typename F, typename ...A>
bool SPtrXVector<xp<T>>::LacksTruth(F&& Function, A&& ...Args) const
{
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++) {
        if (Function((*it).Get(), std::forward<A>(Args)...))
            return false;
    }
    return true;
}


// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename L>
xvector<xp<T>> SPtrXVector<xp<T>>::GetCommonItems(L& item) 
{
    std::sort(The.begin(), The.end());
    std::sort(item.begin(), item.end());

    xvector<xp<T>> vret(The.size() + item.size());
    set_intersection(The.begin(), The.end(), item.begin(), item.end(), vret.begin());
    return vret;
}

template<typename T>
template<typename L>
xvector<xp<T>> SPtrXVector<xp<T>>::GetCommonItems(L&& item) {
    return The.GetCommonItems(item);
}

template<typename T>
template<typename ...R>
inline void SPtrXVector<xp<T>>::Emplace(R && ...Args)
{
    The.emplace_back(RA::MakeShared<T>(std::forward<R>(Args)...));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline void SPtrXVector<xp<T>>::operator<<(const xp<T>& item)
{
    The.emplace_back(item);
}

template<typename T>
inline void SPtrXVector<xp<T>>::operator<<(xp<T>&& item){
    The.emplace_back(std::move(item));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline T& SPtrXVector<xp<T>>::First(size_t Idx)
{
    if (!The.HasIndex(Idx))
        throw "Size is Zero";
    return The.operator[](Idx);
}

template<typename T>
inline const T& SPtrXVector<xp<T>>::First(size_t Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Size is Zero";
    return The.operator[](Idx);
}

template<typename T>
inline T& SPtrXVector<xp<T>>::Last(size_t Idx)
{
    if (!The.HasIndex(Idx))
        throw "Size is Zero";
    return The.operator[](The.size() - Idx - 1);
}


template<typename T>
inline const T& SPtrXVector<xp<T>>::Last(size_t Idx) const
{
    if (!The.HasIndex(Idx))
        throw "Size is Zero";
    return The.operator[](The.size() - Idx - 1);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename C>
inline xvector<C> SPtrXVector<xp<T>>::Convert() const
{
    xvector<C> ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        ret << C((*it).Get());
    return ret;
}


template<typename T>
template<typename C, typename F>
inline xvector<C> SPtrXVector<xp<T>>::Convert(F function) const
{
    xvector<C> ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        ret << function((*it).Get());
    return ret;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
void SPtrXVector<xp<T>>::operator+=(const xvector<xp<T>>& other)
{
    The.insert(The.end(), other.begin(), other.end());
}

template<typename T>
xvector<xp<T>> SPtrXVector<xp<T>>::operator+(const xvector<xp<T>>& other) const 
{
    xvector<xp<T>> vret;
    vret.reserve(The.size());
    vret.insert(vret.end(),  The.begin(), The.end());
    vret += other;
    return vret;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline size_t SPtrXVector<xp<T>>::Size() const {
    return The.size();
}

template<typename T>
inline void SPtrXVector<xp<T>>::Organize()
{
    std::multiset<xp<T>> set_arr;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        set_arr.insert(*it);

    The.clear();
    The.reserve(set_arr.size());

    for (typename std::multiset<xp<T>>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        The.push_back(*it);
}

template<typename T>
inline void SPtrXVector<xp<T>>::RemoveDups()
{
    std::set<xp<T>> set_arr;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        set_arr.insert(*it);

    The.clear();
    The.reserve(set_arr.size());

    for (typename std::set<xp<T>>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        The.push_back(*it);
}

// -----------------------------------------------------------------------------------------------

template<typename T>
template<typename F>
inline xvector<xp<T>>& SPtrXVector<xp<T>>::Sort(F&& func)
{
    // Step through each element of the LvArray except the last
    const auto LnLeng = The.Size();
    for (xint iteration = 0; iteration < LnLeng - 1; ++iteration)
    {
        // Account for the fact that the last element is already sorted with each subsequent iteration
        // so our FvArray "ends" one element sooner
        const xint LnEndOfArray = LnLeng - iteration;

        bool LbSwapped = false; // Keep track of whether any elements were swapped this iteration

        // Search through all elements up to the end of the FvArray - 1
        // The last element has no pair to compare against
        for (int Idx = 0; Idx < LnEndOfArray - 1; ++Idx)
        {
            // If the current element is larger than the element after it
            if (func(The[Idx], The[Idx + 1])) // Generally testing (Left > Right)
            {
                // Swap them
                std::swap(The[Idx], The[Idx + 1]);
                LbSwapped = true;
            }
        }

        // If we haven't swapped any elements this iteration, we're done early
        if (!LbSwapped)
        {
            // iteration is 0 based, but counting iterations is 1-based.  So add 1 here to adjust.
            break;
        }
    }

    //std::sort(The.begin(), The.end(), func);
    return *reinterpret_cast<xvector<xp<T>>*>(this);
}

template<typename T>
inline xvector<xp<T>>& SPtrXVector<xp<T>>::Sort()
{
    std::sort(The.begin(), The.end());
    return *reinterpret_cast<xvector<xp<T>>*>(this);
}

template<typename T>
template<typename F>
inline xvector<xp<T>>& SPtrXVector<xp<T>>::ReverseSort(F&& func)
{
    const auto LnLeng = The.Size();
    for (xint iteration = LnLeng - 1; iteration > 0; iteration--)
    {
        const xint LnEndOfArray = LnLeng - iteration;
        bool LbSwapped = false;
        for (xint Idx = LnEndOfArray - 1; Idx > 0 - 1; Idx--)
        {
            if (func(*The[Idx], *The[Idx + 1])) // Generally testing (Left < Right)
            {
                std::swap(The[Idx], The[Idx + 1]);
                LbSwapped = true;
            }
        }

        if (!LbSwapped)
        {
            std::cout << "Early termination on iteration: " << iteration + 1 << '\n';
            break;
        }
    }
    return *reinterpret_cast<xvector<xp<T>>*>(this);
}

template<typename T>
inline xvector<xp<T>>& SPtrXVector<xp<T>>::ReverseIt()
{
    std::reverse(The.begin(), The.end());
    return *reinterpret_cast<xvector<xp<T>>*>(this);
}

template<typename T>
template<typename F, typename... A>
inline void SPtrXVector<xp<T>>::Proc(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++) {
        if (function((*it).Get(), Args...))
            break;
    }
}

template<typename T>
template<typename F, typename... A>
inline void SPtrXVector<xp<T>>::ProcThread(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
        Nexus<>::AddJobVal(function, (*it).Get(), std::ref(Args)...);
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> SPtrXVector<xp<T>>::ForEach(F&& function, A&& ...Args) const
{
    xvector<N> vret;
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++)
        vret.push_back(function((*it).Get(), Args...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> SPtrXVector<xp<T>>::ForEach(F&& function, A&& ...Args)
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
        rmap.insert(function((*it).Get(), Args...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> SPtrXVector<xp<T>>::ForEach(F&& function, A&& ...Args)
{
    xvector<N> vret;
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
        vret.push_back(function((*it).Get(), Args...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> SPtrXVector<xp<T>>::ForEach(F&& function, A&& ...Args) const
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++)
        rmap.insert(function((*it).Get(), Args...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> SPtrXVector<xp<T>>::ForEachThread(F&& function, A&& ...Args)
{
    if constexpr (std::is_same_v<N, T>)
    {
        CheckRenewObj(VectorPool);
        for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
            VectorPool.AddJobVal(function, (*it).Get(), std::ref(Args)...);

        return VectorPool.GetMoveAllIndices();
    }
    else
    {
        Nexus<N> LoNexus;
        for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
            LoNexus.AddJobVal(function, (*it).Get(), std::ref(Args)...);

        return LoNexus.GetMoveAllIndices();
    }
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> SPtrXVector<xp<T>>::ForEachThread(F&& function, A&& ...Args) const
{
    Nexus<N> LoNexus;
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++)
        LoNexus.AddJobVal(function, (*it).Get(), std::ref(Args)...);

    return LoNexus.GetMoveAllIndices();
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> SPtrXVector<xp<T>>::ForEachThread(F&& function, A&& ...Args) const
{
    auto& MapPool = *RA::MakeShared<Nexus<std::pair<K, V>>>();
    for (typename xvector<E>::const_iterator it = The.begin(); it != The.end(); it++)
        MapPool.AddJobVal(function, (*it).Get(), std::ref(Args)...);

    return MapPool.GetMoveAllIndices();
}

template<typename T>
template <typename N, typename F, typename ...A>
inline void SPtrXVector<xp<T>>::StartTasks(F&& function, A&& ...Args)
{
    CheckRenewObj(VectorPool);
    for (typename xvector<E>::iterator it = The.begin(); it != The.end(); it++)
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
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin() + LnModSize; it != The.end(); it++) {
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
        return The[0];

    T LnModSize = 0;
    if (Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 1;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin() + LnModSize; it != The.end(); it++) {
        num *= (*it).Get();
    }
    return num;
}

template<typename T>
inline T SPtrXVector<xp<T>>::GetAvg(size_t FnSkipIdx) const
{
    return The.GetSum(FnSkipIdx) / (The.Size() - FnSkipIdx);
}

// =============================================================================================================

template<typename T>
inline T SPtrXVector<xp<T>>::Join(const T& str) const
{
    T ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += (*it).Get() + str;

    return ret.substr(0, ret.length() - str.size());
}

template<typename T>
T SPtrXVector<xp<T>>::Join(const char str) const
{
    T ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += (*it).Get() + str;

    return ret.substr(0, ret.length() - 1);
}

template<typename T>
T SPtrXVector<xp<T>>::Join(const char* str) const
{
    T ret;
    for (typename SPtrXVector<xp<T>>::const_iterator it = The.begin(); it != The.end(); it++)
        ret += (*it).Get() + str;

    return ret.substr(0, ret.length() - strlen(str));
}

// =============================================================================================================

#ifndef UsingNVCC
template<typename T>
inline bool SPtrXVector<xp<T>>::FullMatchOne(const re2::RE2& in_pattern) const
{
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (RE2::FullMatch(**iter, in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
inline bool SPtrXVector<xp<T>>::FullMatchAll(const re2::RE2& in_pattern) const
{
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!RE2::FullMatch(**iter, in_pattern)) {
            return false;
        }
    }
    return true;
}


template<typename T>
inline bool SPtrXVector<xp<T>>::MatchOne(const re2::RE2& in_pattern) const
{
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (RE2::PartialMatch(**iter, in_pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
inline bool SPtrXVector<xp<T>>::MatchAll(const re2::RE2& in_pattern) const
{
    for (typename xvector<T*>::const_iterator iter = The.begin(); iter != The.end(); iter++) {
        if (!RE2::PartialMatch(**iter, in_pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Take(const re2::RE2& in_pattern) const
{
    xvector<T*> RetVec;
    RetVec.reserve(The.size() + 1);
    for (size_t i = 0; i < The.size(); i++) {
        if ((RE2::PartialMatch(*(*this)[i], in_pattern)))
            RetVec.push_back((*this)[i]);
    }
    return RetVec;
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Remove(const re2::RE2& in_pattern) const
{
    xvector<T*> RetVec;
    RetVec.reserve(The.size() + 1);
    for (size_t i = 0; i < The.size(); i++) {
        if (!(RE2::PartialMatch(*(*this)[i], in_pattern)))
            RetVec.push_back((*this)[i]);
    }
    return RetVec;
}

template<typename T>
inline xvector<T> SPtrXVector<xp<T>>::SubAll(const re2::RE2& in_pattern, const std::string& replacement) const
{
    xvector<E> RetVec;
    RetVec.reserve(The.size() + 1);
    for (const T* Val : *this)
        RetVec << *Val;

    for (typename SPtrXVector<xp<T>>::iterator iter = RetVec.begin(); iter != RetVec.end(); iter++)
        RE2::GlobalReplace(&*iter, in_pattern, replacement.c_str());
    return RetVec;
}

#endif

// =============================================================================================================


template<typename T>
bool SPtrXVector<xp<T>>::FullMatchOne(const std::string& in_pattern) const
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
bool SPtrXVector<xp<T>>::FullMatchOne(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.FullMatchOne(std::string(in_pattern));
#else
    return The.FullMatchOne(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool SPtrXVector<xp<T>>::FullMatchAll(const std::string& in_pattern) const
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
bool SPtrXVector<xp<T>>::FullMatchAll(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.FullMatchAll(std::string(in_pattern));
#else
    return The.FullMatchAll(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool SPtrXVector<xp<T>>::MatchOne(const std::string& in_pattern) const
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
bool SPtrXVector<xp<T>>::MatchOne(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.MatchOne(std::string(in_pattern));
#else
    return The.MatchOne(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
bool SPtrXVector<xp<T>>::MatchAll(const std::string& in_pattern) const {
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
bool SPtrXVector<xp<T>>::MatchAll(char const* in_pattern) const {
#ifdef UsingNVCC
    return The.MatchAll(std::string(in_pattern));
#else
    return The.MatchAll(re2::RE2(in_pattern));
#endif
}
// =============================================================================================================

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Take(const std::string& in_pattern) const
{
#ifdef UsingNVCC
    xvector<xp<T>> RetVec;
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
inline xvector<xp<T>> SPtrXVector<xp<T>>::Take(char const* in_pattern) const
{
#ifdef UsingNVCC
    return The.Take(std::string(in_pattern));
#else
    return The.Take(re2::RE2(in_pattern));
#endif
}

template<typename T>
inline xvector<xp<T>> SPtrXVector<xp<T>>::Remove(const std::string& in_pattern) const
{
#ifdef UsingNVCC
    xvector<xp<T>> RetVec;
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
inline xvector<xp<T>> SPtrXVector<xp<T>>::Remove(char const* in_pattern) const
{
#ifdef UsingNVCC
    return The.Remove(std::string(in_pattern));
#else
    return The.Remove(re2::RE2(in_pattern));
#endif
}

// =============================================================================================================

template<typename T>
inline xvector<T> SPtrXVector<xp<T>>::SubAll(const std::string& Pattern, const std::string& Replacement) const
{
#ifdef UsingNVCC
    xvector<T> RetVec = The;
    std::regex rex(Pattern, RXM::ECMAScript);
    for (typename SPtrXVector<xp<T>>::iterator iter = RetVec.begin(); iter != RetVec.end(); iter++)
        RetVec.push_back(std::regex_replace(**iter, rex, Replacement));
    return RetVec;
#else
    return The.SubAll(re2::RE2(Pattern.c_str()), Replacement);
#endif
}

template<typename T>
inline xvector<T> SPtrXVector<xp<T>>::SubAll(char const* Pattern, char const* Replacement) const
{
    return The.SubAll(Pattern, Replacement);
}

template<typename T>
inline void SPtrXVector<xp<T>>::InSubAll(const xp<T>& Pattern, const xp<T>& Replacement)
{
    for (xp<T>& Ptr : The)
    {
        if (Ptr.Get() == Pattern.Get())
            Ptr = Replacement;
    }
}

template<typename T>
inline void SPtrXVector<xp<T>>::InSubOne(const xp<T>& Pattern, const xp<T>& Replacement)
{
    for (xp<T>& Ptr : The)
    {
        if (Ptr.Get() == Pattern.Get())
        {
            Ptr = Replacement;
            return;
        }
    }
}

// =============================================================================================================

template<typename T>
xvector<xp<T>> SPtrXVector<xp<T>>::operator()(long double x, long double y, long double z, const char removal_method) const {

    size_t m_size = The.size();
    xvector<xp<T>> n_arr;
    n_arr.reserve(m_size + 4);

    double n_arr_size = static_cast<double>(m_size) - 1;

    if (z >= 0) {

        if (x < 0) { x += n_arr_size; }

        if (!y) { y = n_arr_size; }
        else if (y < 0) { y += n_arr_size; }
        ++y;

        if (x > y) { return n_arr; }

        typename SPtrXVector<xp<T>>::const_iterator iter = The.begin();
        typename SPtrXVector<xp<T>>::const_iterator stop = The.begin() + static_cast<size_t>(y);

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
        
        typename SPtrXVector<xp<T>>::const_reverse_iterator iter = The.rend() - static_cast<size_t>(x) - 1;
        typename SPtrXVector<xp<T>>::const_reverse_iterator stop = The.rend() - static_cast<size_t>(y);

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
