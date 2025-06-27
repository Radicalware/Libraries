

// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]
#if UsingMSVC
#include "Omaha.h"
#else
#include "Omaha.cuh"
#endif
#include "xstring.h"
#include "Macros.h"
#include "Normals.h"

RA::Omaha::Omaha(
    const EHardware FeHardware,
    const double* FvValues,
    const   xint* FnInsertIdxPtr,
    const   xdbl* FnMinPtr,
    const   xdbl* FnMaxPtr,
    const   xint  FnStorageSize) 
    :
    MeHardware(FeHardware),
    MvValues(FvValues),
    MnInsertIdxPtr(FnInsertIdxPtr),
    MnMinPtr(FnMinPtr),
    MnMaxPtr(FnMaxPtr),
    MnStorageSize(FnStorageSize)
{
    
}

RA::Omaha::~Omaha()
{

}

DXF void RA::Omaha::CopyStats(const Omaha& Other)
{
    if (this == &Other)
        return;

    MeHardware = Other.MeHardware;
    MvValues = Other.MvValues;
    MnMinPtr = Other.MnMinPtr;
    MnMaxPtr = Other.MnMaxPtr;
    MvTimeseries = Other.MvTimeseries;
    MmDblToIndex = Other.MmDblToIndex;
    MvSet = Other.MvSet;
    MnInsertIdxPtr = Other.MnInsertIdxPtr;
    MnStorageSize = Other.MnStorageSize;
    MnRunningSize = Other.MnRunningSize;
}

DXF void RA::Omaha::Update()
{
    if (MvValues == nullptr || MnStorageSize == 0)
    {
        printf(RED "Normals needs storage to work\n" WHITE);
        return;
    }

    Insert(MvValues[*MnInsertIdxPtr]);
    if (++MnRunningSize > MnStorageSize)
    {
        MnRunningSize = MnStorageSize;
        RemoveOldest();
    }

    return DXF void();
}

DXF void RA::Omaha::Update(const double FnValue)
{
    ThrowIt("Not Coded Yet");
}

DXF void RA::Omaha::SetDefaultValues(const double FnValue)
{
    MnRunningSize = 0;
    MvTimeseries = {};
    MmDblToIndex = {};
    MvSet = {};
}

void RA::Omaha::Insert(const double FnValue)
{
    MvTimeseries.push_back(FnValue);
    auto It = std::prev(MvTimeseries.end());
    MmDblToIndex.emplace(FnValue, It);
    MvSet.insert(FnValue);
}

void RA::Omaha::Remove(double FnValue)
{
    auto Range = MmDblToIndex.equal_range(FnValue);
    if (Range.first != Range.second) {
        auto It = std::prev(Range.second);
        MvTimeseries.erase(It->second);
        MmDblToIndex.erase(It);
        MvSet.erase(MvSet.find(FnValue));
    }
}

void RA::Omaha::RemoveOldest() 
{
    if (MvTimeseries.empty())
        return;
    cvar OldestValue = MvTimeseries.front();
    auto It = MmDblToIndex.find(OldestValue);
    MvTimeseries.pop_front();
    MmDblToIndex.erase(It);
    MvSet.erase(MvSet.find(OldestValue));
}

xint RA::Omaha::OldIndexFor(double FnValue) const
{
    auto Range = MmDblToIndex.equal_range(FnValue);
    if (Range.first != Range.second) {
        auto It = std::prev(Range.second);
        auto ConstIt = It->second;
        return std::distance<std::list<double>::const_iterator>(
            MvTimeseries.cbegin(), ConstIt);
    }
    return -1;
}

DXF xint RA::Omaha::NewIndexFor(double Value) const {
    for (auto It = MvTimeseries.rbegin(); It != MvTimeseries.rend(); ++It) {
        if (RA::Appx(*It, Value)) {
            return std::distance(MvTimeseries.rbegin(), It);
        }
    }
    return -1;
}

bool RA::Omaha::BxHigh() const 
{
    cvar MaxValue = GetMax();
    xint MaxIndex = OldIndexFor(MaxValue);
    for (auto It = MvTimeseries.rbegin(); It != MvTimeseries.rend(); ++It) {
        if (RA::Appx(*It, MaxValue)) {
            return true;
        }
        else if (RA::Appx(*It, GetMin())) {
            return false;
        }
    }
    return false;
}

bool RA::Omaha::BxLow() const 
{
    cvar MinValue = GetMin();
    xint MinIndex = OldIndexFor(MinValue);
    for (auto It = MvTimeseries.rbegin(); It != MvTimeseries.rend(); ++It) {
        if (RA::Appx(*It, MinValue)) {
            return true;
        }
        else if (RA::Appx(*It, GetMax())) {
            return false;
        }
    }
    return false;
}

DXF double RA::Omaha::GetHighIdxScaled() const
{
    return RA::Normals::ToNormalLinear(
        (double)OldIndexFor(GetMax()), // high index for high size
        RA::Normals::Config(1, 0, MnRunningSize - 1)
    );
}

DXF double RA::Omaha::GetLowIdxScaled() const
{
    return RA::Normals::ToNormalLinear(
        (double)OldIndexFor(GetMin()),
        RA::Normals::Config(1, 0, MnRunningSize - 1)
    );
}
