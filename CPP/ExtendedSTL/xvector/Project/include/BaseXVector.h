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

#ifndef UsingNVCC
#include "re2/re2.h"
#endif


#ifndef __RXM__
#define __RXM__
namespace RXM {
    using namespace std::regex_constants;
    using Type = syntax_option_type;
}
#endif

#include "RawMapping.h"
#include "SharedPtr.h"
#include "Mutex.h"
#include "Nexus.h"

#if (defined(WINDOWS))
using size64_t = __int64;
#else
#include <cstdint>
using size64_t = int64_t;
#endif

// Declared in xvector.h && BaseXVector.h
#ifndef __XVECTOR_TYPES__
#define __XVECTOR_TYPES__
template<typename T, typename enabler_t = void> class xvector;

// Values (Object/Primitive)
#define ValObjXVectorAPI  xvector<T, typename std::enable_if_t<!IsFundamental(T) && !IsPointer(T) && !IsSharedPtr(T)>>
#define ValPrimXVectorAPI xvector<T, typename std::enable_if_t< IsFundamental(T) && !IsPointer(T)>>
template<typename T> class ValObjXVectorAPI;
template<typename T> class ValPrimXVectorAPI;

// Pointers (Object/Primitive)
#define PtrObjXVectorAPI  xvector<T*, typename std::enable_if_t<!IsFundamental(RemovePtr(T*))>>
#define PtrPrimXVectorAPI xvector<T*, typename std::enable_if_t< IsFundamental(RemovePtr(T*))>>
template<typename T> class PtrObjXVectorAPI;
template<typename T> class PtrPrimXVectorAPI;

// Shared Pointers
#define SPtrObjXVectorAPI xvector<xp<T>, typename std::enable_if_t<!IsFundamental(T) && IsSharedPtr(T)>>
template<typename T> class SPtrObjXVectorAPI;
#endif


template<typename T>
class BaseXVector : public std::vector<T, std::allocator<T>>, public RA::MutexHandler
{
public:

    using E = typename std::remove_const<T>::type; // E for Erratic
    using value_type = T;

    ~BaseXVector();
    using std::vector<T, std::allocator<T>>::vector;
    using std::vector<T, std::allocator<T>>::operator=;

    // constexpr void operator=(std::initializer_list<T>&& Other)   { std::vector<T>::operator=(std::move(Other)); };

    constexpr BaseXVector() {}
    constexpr BaseXVector(const std::vector<T, std::allocator<T>>& Other) : std::vector<T>(Other) { }
    constexpr BaseXVector(std::vector<T, std::allocator<T>>&& Other) noexcept : std::vector<T>(std::move(Other)) { };
    constexpr BaseXVector(const xvector<T>& Other) : std::vector<T>(Other) { };
    constexpr BaseXVector(xvector<T>&& Other) noexcept : std::vector<T>(std::move(Other)) { };

    template<typename O> constexpr BaseXVector(const std::vector<O, std::allocator<O>>& Other);
    template<typename O> constexpr BaseXVector(std::vector<O, std::allocator<O>>&& Other) noexcept;
    template<typename O> constexpr BaseXVector(const xvector<O>& Other);
    template<typename O> constexpr BaseXVector(xvector<O>&& Other) noexcept;


    //constexpr void operator=(const std::vector<T, std::allocator<T>>&  Other) { std::vector<T>::operator=(Other); };
    //constexpr void operator=(      std::vector<T, std::allocator<T>>&& Other) { std::vector<T>::operator=(std::move(Other)); };
    constexpr void operator=(const xvector<T>& Other) { std::vector<T>::operator=(Other); };
    constexpr void operator=(xvector<T>&& Other) noexcept { std::vector<T>::operator=(std::move(Other)); };

    constexpr bool operator!(void) const;

    constexpr size_t Size()   const { return The.size(); }
    constexpr size_t Length() const { return The.size(); }
    constexpr bool   Empty()  const { return The.size() == 0; }
              T*     Ptr()          { return The.data(); }
        const T*     Ptr()    const { return The.data(); }
    constexpr void   Reserve(size_t FnNewSize) { The.reserve(FnNewSize); }
    constexpr void   Resize (size_t FnNewSize) { The.resize (FnNewSize); }

    constexpr bool HasRange(const size_t FnSize) const { return Size() >= FnSize; }
    constexpr bool HasIndex(const size_t FnSize) const { return Size() > 0 && Size() - 1 >= FnSize; }

    constexpr void operator*=(const size_t count);

    constexpr void Add(T&& val);
    template<typename ...R>
    constexpr void Add(T&& FtArgs, R... FvRest);
    constexpr void Add(const T& Item);

    constexpr void EraseAll() { The.erase(The.begin(), The.end()); }

    template<typename O>
    constexpr bool operator>(const O& other) const;
    template<typename O>
    constexpr bool operator<(const O& other) const;
    template<typename O>
    constexpr bool operator==(const O& other) const;
    template<typename O>
    constexpr bool operator!=(const O& other) const;

    constexpr bool operator>(const size_t value) const;
    constexpr bool operator<(const size_t value) const;
    constexpr bool operator==(const size_t value) const;
    constexpr bool operator!=(const size_t value) const;

    constexpr xvector<xvector<T>> Split(size_t FnSplinters) const;
    void RemoveLast() { The.pop_back(); }

protected:
    Nexus<std::remove_pointer_t<E>>* VectorPoolPtr = nullptr;
};

template<typename T>
BaseXVector<T>::~BaseXVector()
{
    HostDelete(VectorPoolPtr);
}


template<typename T>
template<typename O>
constexpr BaseXVector<T>::BaseXVector(const std::vector<O, std::allocator<O>>& Other)
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), Other.begin(), Other.end());
}
template<typename T>
template<typename O>
constexpr BaseXVector<T>::BaseXVector(std::vector<O, std::allocator<O>>&& Other) noexcept
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
}
template<typename T>
template<typename O>
constexpr BaseXVector<T>::BaseXVector(const xvector<O>& Other)
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), Other.begin(), Other.end());
}
template<typename T>
template<typename O>
constexpr BaseXVector<T>::BaseXVector(xvector<O>&& Other) noexcept
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline constexpr bool BaseXVector<T>::operator!(void) const
{
    return The.size() == 0;
}

template<typename T>
constexpr void BaseXVector<T>::operator*=(const size_t count)
{
    xvector<T>* tmp = new xvector<T>;
    tmp->reserve(The.size() * count + 1);
    for (int i = 0; i < count; i++)
        The.insert(The.end(), tmp->begin(), tmp->end());
    delete tmp;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
constexpr void BaseXVector<T>::Add(T&& val)
{
    The.emplace_back(std::forward<T>(val));
}
template<typename T>
template<typename ...R>
constexpr void BaseXVector<T>::Add(T&& FtArgs, R... FvRest)
{
    The.emplace_back(FtArgs);
    The.Add(std::forward<R>(FvRest)...);
}

template<typename T>
inline constexpr void BaseXVector<T>::Add(const T& Item)
{
    The.emplace_back(Item);
}

// ------------------------------------------------------------------------------------------------
template<typename T>
template<typename O>
constexpr bool BaseXVector<T>::operator>(const O& other) const
{
    return The.size() > other.size();
}

template<typename T>
template<typename O>
constexpr bool BaseXVector<T>::operator<(const O& other) const
{
    return The.size() < other.size();
}

template<typename T>
template<typename O>
constexpr bool BaseXVector<T>::operator==(const O& other) const
{
    for (T* it : other) {
        if (The.Lacks(it))
            return false;
    }
    return true;
}

template<typename T>
template<typename O>
constexpr bool BaseXVector<T>::operator!=(const O& other) const
{
    for (T* it : other) {
        if (The.Lacks(it))
            return true;
    }
    return false;
}
// --------------------------------------------------------
template<typename T>
constexpr bool BaseXVector<T>::operator>(const size_t value) const
{
    return The.size() > value;
}

template<typename T>
constexpr bool BaseXVector<T>::operator<(const size_t value) const
{
    return The.size() < value;
}

template<typename T>
constexpr bool BaseXVector<T>::operator==(const size_t value) const
{
    return The.size() == value;
}

template<typename T>
constexpr bool BaseXVector<T>::operator!=(const size_t value) const
{
    return The.size() != value;
}

template<typename T>
constexpr xvector<xvector<T>> BaseXVector<T>::Split(size_t FnSplinters) const
{
    xvector<xvector<T>> RetVec;
    if (!The.size() || FnSplinters <= 1)
        return RetVec;

    FnSplinters--; // because index is always 1 less than Count
    if (FnSplinters > The.size())
        FnSplinters = The.size() - 1;

    size_t IdxNum = 0;
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
