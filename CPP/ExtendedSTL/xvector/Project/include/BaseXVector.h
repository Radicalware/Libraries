#pragma once

#include<vector>
#include<utility>
#include<type_traits>
#include<initializer_list>
#include<string>
#include<regex>
#include<sstream>
#include<set>
#include<type_traits>

#include "XvectorTypes.h"
#include "re2/re2.h"
#include "RawMapping.h"
#include "SharedPtr.h"
#include "Mutex.h"
#include "Nexus.h"


template<typename T>
class BaseXVector : public std::vector<T, std::allocator<T>>, public RA::MutexHandler
{
public:

    using E = typename std::remove_const<T>::type; // E for Erratic
    using value_type = T;

    using std::vector<T, std::allocator<T>>::vector;
    using std::vector<T, std::allocator<T>>::operator=;

    // void operator=(std::initializer_list<T>&& Other)   { std::vector<T>::operator=(std::move(Other)); };

    CIN BaseXVector() {}
    CIN BaseXVector(const std::vector<T, std::allocator<T>>&  Other) :          std::vector<T>(Other) { }
    CIN BaseXVector(      std::vector<T, std::allocator<T>>&& Other) noexcept : std::vector<T>(std::move(Other)) { };
    CIN BaseXVector(const BaseXVector<T>&  Other) : std::vector<T>(Other) { };
    CIN BaseXVector(      BaseXVector<T>&& Other) noexcept : std::vector<T>(std::move(Other)) { };

    CIN void operator=(const std::vector<T, std::allocator<T>>&  Other) { std::vector<T>::operator=(Other); };
    CIN void operator=(      std::vector<T, std::allocator<T>>&& Other) { std::vector<T>::operator=(std::move(Other)); };
    CIN void operator=(const BaseXVector<T>&  Other) { std::vector<T>::operator=(Other); };
    CIN void operator=(      BaseXVector<T>&& Other) noexcept { std::vector<T>::operator=(std::move(Other)); };

    template<typename O> CIN BaseXVector(const std::vector<O, std::allocator<O>>&  Other);
    template<typename O> CIN BaseXVector(      std::vector<O, std::allocator<O>>&& Other) noexcept;
    template<typename O> CIN BaseXVector(const BaseXVector<O>&  Other);
    template<typename O> CIN BaseXVector(      BaseXVector<O>&& Other) noexcept;

    template<typename O> CIN void operator=(const std::vector<O, std::allocator<O>>&  Other);
    template<typename O> CIN void operator=(      std::vector<O, std::allocator<O>>&& Other) noexcept;
    template<typename O> CIN void operator=(const BaseXVector<O>&  Other);
    template<typename O> CIN void operator=(      BaseXVector<O>&& Other) noexcept;

    RIN     bool operator!(void) const;

    RIN     xint Size()   const { return The.size(); }
    RIN     xint Length() const { return The.size(); }
    RIN     bool Empty()  const { return The.size() == 0; }
            T*   Ptr()          { return The.data(); }
    RIN CST T*   Ptr()    const { return The.data(); }
    RIN     void Reserve(xint FnNewSize) { The.reserve(FnNewSize); }
    RIN     void Resize (xint FnNewSize) { The.resize (FnNewSize); }

    RIN     bool HasRange(const xint FnSize) const { return Size() >= FnSize; }
    RIN     bool HasIndex(const xint FnSize) const { return Size() > 0 && Size() - 1 >= FnSize; }

    RIN     void operator*=(const xint count);

    template<typename ...R>
    RIN     void Add(T&& FtArgs, R... FvRest);
    RIN     void Add(T&& val);
    RIN     void Add(const T& Item);

    RIN     void EraseAll() { The.erase(The.begin(), The.end()); }

    template<typename O> RIN bool operator> (const O& other) const;
    template<typename O> RIN bool operator< (const O& other) const;
    template<typename O> RIN bool operator==(const O& other) const;
    template<typename O> RIN bool operator!=(const O& other) const;

    RIN bool operator> (const xint value) const;
    RIN bool operator< (const xint value) const;
    RIN bool operator==(const xint value) const;
    RIN bool operator!=(const xint value) const;

    xvector<xvector<T>> RIN Split(xint FnSplinters) const;

    RIN void Remove(const xint Idx);
    RIN void RemoveLast() { The.pop_back(); }

    RIN void ResizeToIdx(const xint FSize) { The.resize(FSize + 1); }
    //RIN void Resize     (const xint FSize) { The.resize(FSize); }

    RIN void ResizeToIdxAndSetAll(const xint FSize, const T& Val);
    RIN void ResizeAndSetAll(const xint FSize, const T& Val);

    template<typename N = T, typename F> RIN xvector<N> ForEachThreadSeq(F&& FfFunction) const;
    template<typename N = T, typename F> RIN xvector<N> ForEachThreadUnseq(F&& FfFunction) const;

    template<typename N = T, typename F> RIN xvector<N> ForEachThreadSeq(F&& FfFunction);
    template<typename N = T, typename F> RIN xvector<N> ForEachThreadUnseq(F&& FfFunction);


    template<typename F> RIN void LoopAllUnseq(F&& FfFunction) const;
    template<typename F> RIN void LoopAllSeq(F&& FfFunction) const;

    template<typename F> RIN void LoopAllUnseq(F&& FfFunction);
    template<typename F> RIN void LoopAllSeq(F&& FfFunction);
};

// -----------------------------------------------------------------------------------------------------------------

template<typename T>
template<typename O>
CIN BaseXVector<T>::BaseXVector(const std::vector<O, std::allocator<O>>& Other)
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), Other.begin(), Other.end());
}
template<typename T>
template<typename O>
CIN BaseXVector<T>::BaseXVector(std::vector<O, std::allocator<O>>&& Other) noexcept
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
    Other.clear();
}
template<typename T>
template<typename O>
CIN BaseXVector<T>::BaseXVector(const BaseXVector<O>& Other)
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), Other.begin(), Other.end());
}
template<typename T>
template<typename O>
CIN BaseXVector<T>::BaseXVector(BaseXVector<O>&& Other) noexcept
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
    Other.clear();
}


// -----------------------------------------------------------------------------------------------------------------

template<typename T>
template<typename O>
CIN void BaseXVector<T>::operator=(const std::vector<O, std::allocator<O>>& Other)
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), Other.begin(), Other.end());
}
template<typename T>
template<typename O>
CIN void BaseXVector<T>::operator=(std::vector<O, std::allocator<O>>&& Other) noexcept
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
    Other.clear();
}
template<typename T>
template<typename O>
CIN void BaseXVector<T>::operator=(const BaseXVector<O>& Other)
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), Other.begin(), Other.end());
}
template<typename T>
template<typename O>
CIN void BaseXVector<T>::operator=(BaseXVector<O>&& Other) noexcept
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
    Other.clear();
}

// -----------------------------------------------------------------------------------------------------------------

template<typename T>
RIN bool BaseXVector<T>::operator!(void) const
{
    return The.size() == 0;
}

template<typename T>
RIN void BaseXVector<T>::operator*=(const xint count)
{
    xvector<T>* tmp = new xvector<T>;
    tmp->reserve(The.size() * count + 1);
    for (int i = 0; i < count; i++)
        The.insert(The.end(), tmp->begin(), tmp->end());
    delete tmp;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename ...R>
RIN void BaseXVector<T>::Add(T&& FtArgs, R... FvRest)
{
    The.emplace_back(FtArgs);
    The.Add(std::forward<R>(FvRest)...);
}

template<typename T>
RIN void BaseXVector<T>::Add(T&& val)
{
    The.emplace_back(std::forward<T>(val));
}

template<typename T>
RIN void BaseXVector<T>::Add(const T& Item)
{
    The.emplace_back(Item);
}

// ------------------------------------------------------------------------------------------------
template<typename T>
template<typename O>
bool RIN BaseXVector<T>::operator>(const O& other) const
{
    return The.size() > other.size();
}

template<typename T>
template<typename O>
bool RIN BaseXVector<T>::operator<(const O& other) const
{
    return The.size() < other.size();
}

template<typename T>
template<typename O>
bool RIN BaseXVector<T>::operator==(const O& other) const
{
    for (T* it : other) {
        if (The.Lacks(it))
            return false;
    }
    return true;
}

template<typename T>
template<typename O>
bool RIN BaseXVector<T>::operator!=(const O& other) const
{
    for (T* it : other) {
        if (The.Lacks(it))
            return true;
    }
    return false;
}
// --------------------------------------------------------
template<typename T>
bool RIN BaseXVector<T>::operator>(const xint value) const
{
    return The.size() > value;
}

template<typename T>
bool RIN BaseXVector<T>::operator<(const xint value) const
{
    return The.size() < value;
}

template<typename T>
bool RIN BaseXVector<T>::operator==(const xint value) const
{
    return The.size() == value;
}

template<typename T>
bool RIN BaseXVector<T>::operator!=(const xint value) const
{
    return The.size() != value;
}

template<typename T>
xvector<xvector<T>> RIN BaseXVector<T>::Split(xint FnSplinters) const
{
    xvector<xvector<T>> RetVec;
    if (!The.size() || FnSplinters <= 1)
        return RetVec;

    FnSplinters--; // because index is always 1 less than Count
    if (FnSplinters > The.size())
        FnSplinters = The.size() - 1;

    xint IdxNum = 0;
    for (auto& Val : The)
    {
        if (IdxNum >= RetVec.size())
            RetVec.push_back(xvector<T>{ Val });
        else
            RetVec[IdxNum].push_back(Val);

        IdxNum++;
        if (IdxNum >= FnSplinters)
            IdxNum = 0;
    }

    return RetVec;
}

template<typename T>
RIN void BaseXVector<T>::Remove(const xint Idx)
{
    if (Idx >= The.size())
        return;
    The.erase(The.begin() + Idx);
}

template<typename T>
RIN void BaseXVector<T>::ResizeToIdxAndSetAll(const xint FSize, const T& Val)
{
    The.resize(FSize + 1);
    for (auto& Elem : The)
        Elem = Val;
}

template<typename T>
RIN void BaseXVector<T>::ResizeAndSetAll(const xint FSize, const T& Val)
{
    The.resize(FSize);
    for (auto& Elem : The)
        Elem = Val;
}

template<typename T>
template<typename N, typename F>
RIN xvector<N> BaseXVector<T>::ForEachThreadSeq(F&& FfFunction) const
{
    xvector<N> LvRet;
    LvRet.Resize(The.size() + 1);

    std::transform(
        std::execution::par,
        The.cbegin(), The.cend(),
        LvRet.begin(),
        FfFunction);

    return LvRet;
}

template<typename T>
template<typename N, typename F>
RIN xvector<N> BaseXVector<T>::ForEachThreadUnseq(F&& FfFunction) const
{
    xvector<N> LvRet;
    LvRet.Resize(The.size() + 1);

    std::transform(
        std::execution::par_unseq,
        The.cbegin(), The.cend(),
        LvRet.begin(),
        FfFunction);

    return LvRet;
}

template<typename T>
template<typename N, typename F>
RIN xvector<N> BaseXVector<T>::ForEachThreadSeq(F&& FfFunction)
{
    xvector<N> LvRet;
    LvRet.Resize(The.size() + 1);

    std::transform(
        std::execution::par,
        The.begin(), The.end(),
        LvRet.begin(),
        FfFunction);

    return LvRet;
}

template<typename T>
template<typename N, typename F>
RIN xvector<N> BaseXVector<T>::ForEachThreadUnseq(F&& FfFunction)
{
    xvector<N> LvRet;
    LvRet.Resize(The.size() + 1);

    std::transform(
        std::execution::par_unseq,
        The.begin(), The.end(),
        LvRet.begin(),
        FfFunction);

    return LvRet;
}

template<typename T>
template<typename F>
RIN void BaseXVector<T>::LoopAllUnseq(F&& FfFunction) const
{
    std::for_each(
        std::execution::par_unseq,
        The.cbegin(),
        The.cend(),
        FfFunction);
}

template<typename T>
template<typename F>
RIN void BaseXVector<T>::LoopAllSeq(F&& FfFunction) const
{
    std::for_each(
        std::execution::par,
        The.cbegin(),
        The.cend(),
        FfFunction);
}

template<typename T>
template<typename F>
RIN void BaseXVector<T>::LoopAllUnseq(F&& FfFunction)
{
    std::for_each(
        std::execution::par_unseq,
        The.begin(),
        The.end(),
        FfFunction);
}

template<typename T>
template<typename F>
RIN void BaseXVector<T>::LoopAllSeq(F&& FfFunction)
{
    std::for_each(
        std::execution::par, 
        The.begin(), 
        The.end(), 
        FfFunction);
}