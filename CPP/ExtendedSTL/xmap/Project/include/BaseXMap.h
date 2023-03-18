#pragma once

#include "Mutex.h"
#include "xvector.h"
#include <unordered_map>
#include <map>

#ifdef CXX20
#define ValueMustCopy
#else
#define ValueMustCopy requires std::is_copy_constructible_v<V>
#endif

#undef  ConstructorHeader
#define ConstructorHeader template<typename K, typename V, typename H> CIN BaseXMap<K, V, H>::BaseXMap


template<typename K, typename V, typename H = std::hash<K>> class xmap;

template<typename K, typename V, typename H>
class BaseXMap : public std::unordered_map<K, V, H>, public RA::MutexHandler
{
public:
    inline  BaseXMap() {}
    inline ~BaseXMap() {}

    using std::unordered_map<K, V, H>::unordered_map;
    using std::unordered_map<K, V, H>::operator=;

    // --------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    CIN BaseXMap(const BaseXMap<K, V, H>&  Other);
    CIN BaseXMap(      BaseXMap<K, V, H>&& Other) noexcept;

    CIN void operator=(const BaseXMap<K, V, H>&  Other);
    CIN void operator=(      BaseXMap<K, V, H>&& Other) noexcept;
    // -------------------------------------------------------------------------
    CIN BaseXMap(const std::unordered_map<K, V>&  Other);
    CIN BaseXMap(      std::unordered_map<K, V>&& Other) noexcept;
    
    CIN void operator=(const std::unordered_map<K, V, H>&  Other);
    CIN void operator=(      std::unordered_map<K, V, H>&& Other) noexcept;
    // -------------------------------------------------------------------------
    CIN BaseXMap(const std::map<K, V>&  Other);
    CIN BaseXMap(      std::map<K, V>&& Other) noexcept;
    
    CIN void operator=(const std::map<K, V, H>&  Other);
    CIN void operator=(      std::map<K, V, H>&& Other) noexcept;
    // -------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    RIN void AddPairIfNotExist(const K& one, const V& two);
    RIN void AddPairIfNotExist(const K& one, const V&& two);

    RIN bool operator!(void) const;

    RIN void RemoveAll();
    RIN xmap<V, K> GetInverted() const;

    template<typename O> RIN void Clone(const O& Other);
    template<typename O> RIN void Clone(      O&& Other) noexcept;
};

ConstructorHeader(const BaseXMap<K, V, H>&  Other)                 : std::unordered_map<K, V, H>(Other.begin(), Other.end()) {  }
ConstructorHeader(      BaseXMap<K, V, H>&& Other)        noexcept : std::unordered_map<K, V, H>(std::move(Other)) {  }

ConstructorHeader(const std::unordered_map<K, V>&  Other)          : std::unordered_map<K, V, H>(Other.begin(), Other.end()) {  }
ConstructorHeader(      std::unordered_map<K, V>&& Other) noexcept : std::unordered_map<K, V, H>(std::move(Other)) {  }

ConstructorHeader(const std::map<K, V>&  Other)                    : std::unordered_map<K, V, H>(Other.begin(), Other.end()){  }
ConstructorHeader(      std::map<K, V>&& Other)           noexcept : std::unordered_map<K, V, H>(std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end())) { Other.clear(); }

// -------------------------------------------------------------------------------------
template<typename K, typename V, typename H>
CIN void BaseXMap<K, V, H>::operator=(const BaseXMap<K, V, H>& Other)
{
    The.clear();
    The.insert(Other.begin(), Other.end());
}

template<typename K, typename V, typename H>
CIN void BaseXMap<K, V, H>::operator=(BaseXMap<K, V, H>&& Other) noexcept
{
    The.clear();
    The = std::unordered_map<K, V, H>::operator=(std::move(Other));
}
// -------------------------------------------------------------------------------------
template<typename K, typename V, typename H>
CIN void BaseXMap<K, V, H>::operator=(const std::unordered_map<K, V, H>& Other)
{
    The.clear();
    The.insert(Other.begin(), Other.end());
}

template<typename K, typename V, typename H>
CIN void BaseXMap<K, V, H>::operator=(std::unordered_map<K, V, H>&& Other) noexcept
{
    The.clear();
    The = std::move(Other);
}
// -------------------------------------------------------------------------------------
template<typename K, typename V, typename H>
CIN void BaseXMap<K, V, H>::operator=(const std::map<K, V, H>& Other)
{
    The.clear();
    The.insert(Other.begin(), Other.end());
}

template<typename K, typename V, typename H>
CIN void BaseXMap<K, V, H>::operator=(std::map<K, V, H>&& Other) noexcept
{
    The.clear();
    The.insert(std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
}
// -------------------------------------------------------------------------------------


template<typename K, typename V, typename H>
RIN void BaseXMap<K, V, H>::AddPairIfNotExist(const K& one, const V& two)
{
    if (this->find(one) == this->end())
        this->insert(std::make_pair(one, two));
}

template<typename K, typename V, typename H>
RIN void BaseXMap<K, V, H>::AddPairIfNotExist(const K& one, const V&& two)
{
    if (this->find(one) == this->end())
        this->insert(std::make_pair(one, std::move(two)));
}

template<typename K, typename V, typename H>
RIN bool BaseXMap<K, V, H>::operator!(void) const
{
    return The.size() == 0;
}


template<typename K, typename V, typename H>
template<typename O>
RIN void BaseXMap<K, V, H>::Clone(const O& Other)
{
    this->clear();
    this->insert(Other.begin(), Other.end());
}

template<typename K, typename V, typename H>
template<typename O>
RIN void BaseXMap<K, V, H>::Clone(O&& Other) noexcept
{
    this->clear();
    this->insert(std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end()));
    Other.clear();
}

#undef ConstructorHeader