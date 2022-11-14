#pragma once

#include "Mutex.h"
#include "xvector.h"
#include <unordered_map>
#include <map>

template<typename K, typename V, typename H = std::hash<K>> class xmap;

template<typename K, typename V, typename H>
class BaseXMap : public std::unordered_map<K, V, H>, public RA::MutexHandler
{
public:

    inline  BaseXMap() {}
    inline ~BaseXMap() {}

    inline BaseXMap(const BaseXMap<K, V, H>&  Other);
    inline BaseXMap(      BaseXMap<K, V, H>&& Other) noexcept;

    inline BaseXMap(const std::unordered_map<K, V, H>&  Other);
    inline BaseXMap(      std::unordered_map<K, V, H>&& Other) noexcept;

    inline BaseXMap(const std::map<K, V, H>&  Other);
    inline BaseXMap(      std::map<K, V, H>&& Other) noexcept;

    using std::unordered_map<K, V, H>::unordered_map;

    inline void AddPairIfNotExist(const K& one, const V& two);
    inline void AddPairIfNotExist(const K& one, const V&& two);

    bool operator!(void) const;

    void RemoveAll();
    xmap<V, K> GetInverted() const;


    template<typename O> inline void Clone(const O& other);
    template<typename O> inline void Clone(      O&& other) noexcept;
};

template<typename K, typename V, typename H>
inline BaseXMap<K, V, H>::BaseXMap(const BaseXMap<K, V, H>& Other) :
    std::unordered_map<K, V, H>(Other.begin(), Other.end()) {}
template<typename K, typename V, typename H>
inline BaseXMap<K, V, H>::BaseXMap(BaseXMap<K, V, H>&& Other) noexcept :
    std::unordered_map<K, V, H>(std::move(Other)) { }

template<typename K, typename V, typename H>
inline BaseXMap<K, V, H>::BaseXMap(const std::unordered_map<K, V, H>& Other) :
    std::unordered_map<K, V, H>(Other.begin(), Other.end()) {}
template<typename K, typename V, typename H>
inline BaseXMap<K, V, H>::BaseXMap(std::unordered_map<K, V, H>&& Other) noexcept :
    std::unordered_map<K, V, H>(std::move(Other)) { }

template<typename K, typename V, typename H>
inline BaseXMap<K, V, H>::BaseXMap(const std::map<K, V, H>& Other) :
    std::unordered_map<K, V, H>(Other.begin(), Other.end()){}
template<typename K, typename V, typename H>
inline BaseXMap<K, V, H>::BaseXMap(std::map<K, V, H>&& Other) noexcept :
    std::unordered_map<K, V, H>(std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end())) { }


template<typename K, typename V, typename H>
inline void BaseXMap<K, V, H>::AddPairIfNotExist(const K& one, const V& two)
{
    if (this->find(one) == this->end())
        this->insert(std::make_pair(one, two));
}

template<typename K, typename V, typename H>
inline void BaseXMap<K, V, H>::AddPairIfNotExist(const K& one, const V&& two)
{
    if (this->find(one) == this->end())
        this->insert(std::make_pair(one, std::move(two)));
}

template<typename K, typename V, typename H>
inline bool BaseXMap<K, V, H>::operator!(void) const
{
    return The.size() == 0;
}


template<typename K, typename V, typename H>
template<typename O>
inline void BaseXMap<K, V, H>::Clone(const O& other)
{
    this->clear();
    this->insert(other.begin(), other.end());
}

template<typename K, typename V, typename H>
template<typename O>
inline void BaseXMap<K, V, H>::Clone(O&& other) noexcept
{
    this->clear();
    this->insert(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
}