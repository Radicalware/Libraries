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
    class AutoMirror : public BaseMirror<K, V, H, C>
    {
    public:
        AKK void AddElement(KK&& FoElement);
        AKK void RemoveElement(KK&& FoElement);
    };
}

template<typename K, typename V, typename H, typename C>
AKK
inline void RA::AutoMirror::AddElement(KK&& FoElement)
{
    if (!FoElement.size())
        throw "FoElement Size Zero";

    if (The.MmKeyToValue.contains(FoElement))
        return;

    const size_t LnAnimalIdx = The.MmKeyToValue.size();
    The.MmKeyToValue.insert({ FoElement, LnAnimalIdx });
    auto It = The.MmKeyToValue.find(FoElement);
    The.MmValueToPtrKey.insert_or_assign(LnAnimalIdx, &It->first);
}

template<typename K, typename V, typename H, typename C>
AKK
inline void RA::AutoMirror::RemoveElement(KK&& FoElement)
{
    if (!The.MmKeyToValue.contains(FoElement))
        return;

    const auto LnAnimalIdx = The.MmKeyToValue[FoElement];
    const auto LoAnimalIt  = The.MmValueToPtrKey[LnAnimalIdx];
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
