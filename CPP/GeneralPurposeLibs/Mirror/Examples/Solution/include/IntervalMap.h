#pragma once

#include "Mirror.h"
#include "Key.h"

using std::cout;
using std::endl;


template<typename K, typename V>
//class IntervalMap : public RA::Mirror<Key<K>, V, Key<>::Hash, Key<>::Equals> // For unordered_map
class IntervalMap : public RA::Mirror<Key<K>, V, Key<>::Hash, Key<>::Compare> // For map
{
    V McValBegin;
public:
    // constructor associates whole range of K with val
    IntervalMap(const V& val)
        : McValBegin(val)
    {
    }

    void Assign(const K& FnKeyBegin, const K& FnKeyEnd, const V& FcValue)
    {
        if (The.Size() == 0)
        {
            auto LoKey = Key<K>(INT_MIN, FnKeyBegin - 1);
            The.AddKVP(LoKey, McValBegin);
        }
        auto LoKey = Key<K>(FnKeyBegin, FnKeyEnd);
        The.AddKVP(LoKey, FcValue);
    }

    void Close() {
        Assign(GetLastKey().MnUpper + 1, INT_MAX, McValBegin);
    }

    template<typename T>
    const auto& GetValue(const T& FnValue) const {
        return *The.MmKeyToValue.find(FnValue);
    }

    auto& GetFirstKey() const { return (*(  The.MmKeyToValue.cbegin())).first; }
    auto& GetLastKey()  const { return (*(--The.MmKeyToValue.cend()))  .first; }

    void PrintMap() const
    {
        for (auto& [LoKey, LoValue] : The.MmKeyToValue)
            cout << LoValue << " : " << LoKey << endl;
    }
};

