#pragma once

/*
* Copyright[2024][Joel Leagues aka Scourge] under the Apache v2 Licence
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
*/

#include "ConfigMirror.h"

namespace RA
{
    template<typename K, typename V, typename H, typename C>
    class BaseMirror
    {
    public:
        auto begin()  noexcept { return MmKeyToValue.begin(); }
        auto end()    noexcept { return MmKeyToValue.end(); }

        auto cbegin() noexcept { return MmKeyToValue.cbegin(); }
        auto cend()   noexcept { return MmKeyToValue.cend(); }

        auto Size() const { return MmKeyToValue.size(); }

        const V& operator[](const K& Idx) const { return  MmKeyToValue.at(Idx); }
        const K& operator[](const V& Idx) const { return *MmValueToPtrKey.at(Idx); }

        void Replace(const size_t FnIndex, const K& FoReplace);
        void Replace(const K& FoFind, const K& FoReplace);

    protected:
        std::map<K, V, C> MmKeyToValue; // aka KeyToIdx
        std::map<V, const K*> MmValueToPtrKey;
    };
};




template<typename K, typename V, typename H, typename C>
inline void RA::BaseMirror<K, V, H, C>::Replace(const size_t FnIndex, const K& FoReplace)
{
    The.MmKeyToValue.erase(The[FnIndex]);
    The.MmValueToPtrKey.erase(FnIndex);
    The.MmKeyToValue.insert_or_assign(FoReplace, FnIndex);
    The.MmValueToPtrKey.insert_or_assign(FnIndex, &The.MmKeyToValue.find(FoReplace)->first);
}

template<typename K, typename V, typename H, typename C>
inline void RA::BaseMirror<K, V, H, C>::Replace(const  K& FoFind, const K& FoReplace)
{
    auto LoIndex = The[FoFind];
    The.MmKeyToValue.erase(FoFind);
    The.MmValueToPtrKey.erase(LoIndex);
    The.MmKeyToValue.insert_or_assign(FoReplace, LoIndex);
    The.MmValueToPtrKey.insert_or_assign(LoIndex, &The.MmKeyToValue.find(FoReplace)->first);
}

