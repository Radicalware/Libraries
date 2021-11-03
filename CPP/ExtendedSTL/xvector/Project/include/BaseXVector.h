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

#include "re2/re2.h"


#include "RawMapping.h"
#include "MutexHandler.h"
#include "Nexus.h"

#if (defined(WINDOWS))
    using size64_t = __int64;
#else
    #include <cstdint>
    using size64_t = int64_t;
#endif

    
template<typename T, typename enabler_t = void> class xvector;
// Values (Object/Primitive)
template<typename T> class xvector<T, typename std::enable_if< std::is_class<T>::value && !std::is_pointer<T>::value>::type>; // val_obj_xvector
template<typename T> class xvector<T, typename std::enable_if<!std::is_class<T>::value && !std::is_pointer<T>::value>::type>; // val_prim_xvector

// Pointers (Object/Primitive)
template<typename T> class xvector<T*, typename std::enable_if< std::is_class<std::remove_pointer_t<T*>>::value>::type>; // ptr_obj_xvector
template<typename T> class xvector<T*, typename std::enable_if<!std::is_class<std::remove_pointer_t<T*>>::value>::type>; // ptr_prim_xvector


template<typename T> 
class BaseXVector : public std::vector<T, std::allocator<T>>, public RA::MutexHandler
{
public:

    using E = std::remove_const<T>::type; // E for Erratic
    using value_type = T;

    inline ~BaseXVector();
    using std::vector<T, std::allocator<T>>::vector;
    using std::vector<T, std::allocator<T>>::operator=;

    // inline void operator=(std::initializer_list<T>&& Other)   { std::vector<T>::operator=(std::move(Other)); };

    inline BaseXVector() {}
    inline BaseXVector(const std::vector<T, std::allocator<T>>& Other)           : std::vector<T>(Other) { }
    inline BaseXVector(      std::vector<T, std::allocator<T>>&& Other) noexcept : std::vector<T>(std::move(Other)) { };
    inline BaseXVector(const xvector<T>&  Other )         : std::vector<T>(Other) { };
    inline BaseXVector(      xvector<T>&& Other) noexcept : std::vector<T>(std::move(Other)) { };

    template<typename O> inline BaseXVector(const std::vector<O, std::allocator<O>>& Other);
    template<typename O> inline BaseXVector(      std::vector<O, std::allocator<O>>&& Other) noexcept;
    template<typename O> inline BaseXVector(const xvector<O>& Other);
    template<typename O> inline BaseXVector(      xvector<O>&& Other) noexcept;


    //inline void operator=(const std::vector<T, std::allocator<T>>&  Other) { std::vector<T>::operator=(Other); };
    //inline void operator=(      std::vector<T, std::allocator<T>>&& Other) { std::vector<T>::operator=(std::move(Other)); };
    inline void operator=(const xvector<T>&  Other) { std::vector<T>::operator=(Other); };
    inline void operator=(      xvector<T>&& Other) noexcept { std::vector<T>::operator=(std::move(Other)); };

    inline size_t Size() const;
    inline bool HasRange(const size_t FnSize) const;
    inline bool HasIndex(const size_t FnSize) const;

    inline void operator*=(const size_t count);

    inline void Add();
    inline void Add(T&& val);
    template <typename Start, typename... Rest>
    inline void Add(Start&& first, Rest&& ... rest);

    template<typename O>
    inline bool operator>(const O& other) const;
    template<typename O>
    inline bool operator<(const O& other) const;
    template<typename O>
    inline bool operator==(const O& other) const;
    template<typename O>
    inline bool operator!=(const O& other) const;

    inline bool operator>(const size_t value) const;
    inline bool operator<(const size_t value) const;
    inline bool operator==(const size_t value) const;
    inline bool operator!=(const size_t value) const;

protected:
    Nexus<std::remove_pointer_t<E>>* VectorPoolPtr = nullptr;
}; 

template<typename T>
inline BaseXVector<T>::~BaseXVector()
{
    Delete(VectorPoolPtr);
}


template<typename T>
template<typename O>
inline BaseXVector<T>::BaseXVector(const std::vector<O, std::allocator<O>>& Other)
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), Other.begin(), Other.end());
}
template<typename T>
template<typename O>
inline BaseXVector<T>::BaseXVector(std::vector<O, std::allocator<O>>&& Other) noexcept
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
}
template<typename T>
template<typename O>
inline BaseXVector<T>::BaseXVector(const xvector<O>& Other)
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), Other.begin(), Other.end());
}
template<typename T>
template<typename O>
inline BaseXVector<T>::BaseXVector(xvector<O>&& Other) noexcept
{
    The.clear();
    The.reserve(Other.size());
    The.insert(The.begin(), std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
}

template<typename T>
inline size_t BaseXVector<T>::Size() const
{
    return this->size();
}


template<typename T>
bool BaseXVector<T>::HasRange(const size_t FnSize) const
{
    if (Size() >= FnSize)
        return true;
    return false;
}


template<typename T>
bool BaseXVector<T>::HasIndex(const size_t FnSize) const
{
    std::vector<int> vals;
    if (Size() - 1 >= FnSize)
        return true;
    return false;
}


// ------------------------------------------------------------------------------------------------

template<typename T>
inline void BaseXVector<T>::operator*=(const size_t count)
{
    xvector<T>* tmp = new xvector<T>;
    tmp->reserve(this->size() * count + 1);
    for (int i = 0; i < count; i++)
        this->insert(this->end(), tmp->begin(), tmp->end());
    delete tmp;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline void BaseXVector<T>::Add() {}
template<typename T>
inline void BaseXVector<T>::Add(T&& val)
{
    this->emplace_back(std::forward<T>(val));
}
template<typename T>
template<typename Start, typename ...Rest>
inline void BaseXVector<T>::Add(Start&& start, Rest&& ...rest)
{
    this->emplace_back(std::forward<Start>(start));
    this->Add(std::forward<Rest>(rest)...);
}

// ------------------------------------------------------------------------------------------------
template<typename T>
template<typename O>
inline bool BaseXVector<T>::operator>(const O& other) const
{
    return this->size() > other.size();
}

template<typename T>
template<typename O>
inline bool BaseXVector<T>::operator<(const O& other) const
{
    return this->size() < other.size();
}

template<typename T>
template<typename O>
inline bool BaseXVector<T>::operator==(const O& other) const
{
    for (T* it : other) {
        if (this->Lacks(it))
            return false;
    }
    return true;
}

template<typename T>
template<typename O>
inline bool BaseXVector<T>::operator!=(const O& other) const
{
    for (T* it : other) {
        if (this->Lacks(it))
            return true;
    }
    return false;
}
// --------------------------------------------------------
template<typename T>
inline bool BaseXVector<T>::operator>(const size_t value) const
{
    return this->size() > value;
}

template<typename T>
inline bool BaseXVector<T>::operator<(const size_t value) const
{
    return this->size() < value;
}

template<typename T>
inline bool BaseXVector<T>::operator==(const size_t value) const
{
    return this->size() == value;
}

template<typename T>
inline bool BaseXVector<T>::operator!=(const size_t value) const
{
    return this->size() != value;
}