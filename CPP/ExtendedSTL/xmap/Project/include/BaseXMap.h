#pragma once

#include "MutexHandler.h"
#include <unordered_map>
#include <map>

template<typename K, typename V>
class BaseXMap : public std::unordered_map<K, V>, public RA::MutexHandler
{
public:

    inline  BaseXMap() {}
    inline ~BaseXMap() {}

    inline BaseXMap(const BaseXMap<K, V>&  Other);
    inline BaseXMap(      BaseXMap<K, V>&& Other) noexcept;

    inline BaseXMap(const std::unordered_map<K, V>&  Other);
    inline BaseXMap(      std::unordered_map<K, V>&& Other) noexcept;

    inline BaseXMap(const std::map<K, V>&  Other);
    inline BaseXMap(      std::map<K, V>&& Other) noexcept;

    using std::unordered_map<K, V>::unordered_map;
    using std::unordered_map<K, V>::operator=;
};


template<typename K, typename V>
inline BaseXMap<K, V>::BaseXMap(const BaseXMap<K, V>& Other) :
    std::unordered_map<K, V>(Other.begin(), Other.end()) {}
template<typename K, typename V>
inline BaseXMap<K, V>::BaseXMap(BaseXMap<K, V>&& Other) noexcept :
    std::unordered_map<K, V>(std::move(Other)) { }

template<typename K, typename V>
inline BaseXMap<K, V>::BaseXMap(const std::unordered_map<K, V>& Other) :
    std::unordered_map<K, V>(Other.begin(), Other.end()) {}
template<typename K, typename V>
inline BaseXMap<K, V>::BaseXMap(std::unordered_map<K, V>&& Other) noexcept :
    std::unordered_map<K, V>(std::move(Other)) { }

template<typename K, typename V>
inline BaseXMap<K, V>::BaseXMap(const std::map<K, V>& Other) :
    std::unordered_map<K, V>(Other.begin(), Other.end()){}
template<typename K, typename V>
inline BaseXMap<K, V>::BaseXMap(std::map<K, V>&& Other) noexcept :
    std::unordered_map<K, V>(std::make_move_iterator(Other.begin()), std::make_move_iterator(Other.end())) { }

