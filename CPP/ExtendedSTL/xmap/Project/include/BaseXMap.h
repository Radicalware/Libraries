#pragma once

#include "MutexHandler.h"
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
    using std::unordered_map<K, V, H>::operator=;

    inline void AddPairIfNotExist(const K& one, const V& two);
    inline void AddPairIfNotExist(const K& one, const V&& two);


    void RemoveAll();
    xmap<V, K> GetInverted() const;
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
inline void BaseXMap<K, V, H>::RemoveAll()
{
    this->erase(this->begin(), this->end());
}

template<typename K, typename V, typename H>
inline xmap<V, K> BaseXMap<K, V, H>::GetInverted() const
{
    xmap<V, K> Ret;
    for (const auto& [Key, Value] : *this)
        Ret.insert({ Value, Key });
    return Ret;
}
