#pragma once

/*
* Copyright[2024][Joel Leagues aka Scourge] under the Apache v2 Licence
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
*/

#include "BaseMirror.h"

namespace RA
{
    template<typename K, typename V, typename H, typename C>
    class ManualMirror : public BaseMirror<K, V, H, C>
    {
    public:
        AKVP void AddKVP(KK&& FoKey, VV&& FoValue);
        AKK void RemoveKey(const KK& FoKey);

        constexpr auto& GetMove(const K& FoKey) noexcept
        {
            if constexpr (std::is_fundamental_v<V>)
                return The[FoKey];
            return std::move(The[FoKey]);
        }
    };
}


template<typename K, typename V, typename H, typename C>
AKVP
inline void RA::ManualMirror::AddKVP(KK&& FoKey, VV&& FoValue)
{
    if (The.MmKeyToValue.contains(FoKey))
    {
        The.MmValueToPtrKey.insert_or_assign(FoValue, &The.MmKeyToValue.find(FoKey)->first);
        return;
    }

    The.MmKeyToValue.insert({ FoKey, FoValue });
    auto It = The.MmKeyToValue.find(FoKey);
    The.MmValueToPtrKey.insert_or_assign(FoValue, &It->first);
}

template<typename K, typename V, typename H, typename C>
AKK
inline void RA::ManualMirror::RemoveKey(const KK& FoKey)
{
    if (!The.MmKeyToValue.contains(FoKey))
        return;

    const auto LnAnimalIdx = The.MmKeyToValue[FoKey];
    const auto LoAnimalIt = The.MmValueToPtrKey[LnAnimalIdx];
    The.MmKeyToValue.erase(*LoAnimalIt);
    if (The.MmKeyToValue.size() > LnAnimalIdx)
    {
        const auto SavedItem = &*The.MmValueToPtrKey.at(The.MmKeyToValue.size());
        The.MmKeyToValue[*SavedItem] = LnAnimalIdx;
        The.MmValueToPtrKey[LnAnimalIdx] = SavedItem;
    }

    auto EndIt = The.MmValueToPtrKey.end();
    --EndIt;
    The.MmValueToPtrKey.erase(EndIt);
}

