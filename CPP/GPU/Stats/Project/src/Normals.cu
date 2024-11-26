
// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]
#if UsingMSVC
#include "Normals.h"
#else
#include "Normals.cuh"
#endif
#include "xstring.h"
#include "Macros.h"

RA::Normals::Normals(
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
    if (FeHardware == EHardware::GPU)
        throw __CLASS__ " >> GPU not coded for this yet";
#ifdef UsingMSVC
    MvNormals.resize(MnStorageSize);
#endif // UsingMSVC

}

RA::Normals::~Normals()
{
}

DXF void RA::Normals::CopyStats(const RA::Normals& Other)
{
    MnLastNormal = Other.MnLastNormal;
}

//DXF double RA::Normals::ToNormal(const double FnRaw, const double FnMin, const double FnMax)
//{
//    if (MbLiteral)
//        return FnRaw;
//    const auto& MnMax = (*MnMaxPtr);
//    const auto& MnMin = (*MnMinPtr);
//    return 2 * ((FnRaw - MnMin) / (MnMax - MnMin)) - 1;
//}
//
//DXF double RA::Normals::ToRaw(const double FnNormal, const double FnMin, const double FnMax)
//{
//    if (MbLiteral)
//        return FnNormal;
//    const auto& MnMax = (*MnMaxPtr);
//    const auto& MnMin = (*MnMinPtr);
//    return ((FnNormal + 1) / 2) * (MnMax - MnMin) + MnMin;
//}
//
//DXF double RA::Normals::ToNormal(const double FnRaw)
//{
//    if (MbLiteral)
//        return FnRaw;
//    const auto& MnMin = (*MnMinPtr);
//    const auto& MnMax = (*MnMaxPtr);
//    return 2 * ((FnRaw - MnMin) / (MnMax - MnMin)) - 1;
//    return Normals::ToNormal(FnRaw, MnMin, MnMax);
//}
//
//DXF double RA::Normals::ToRaw(const double FnNormal)
//{
//    if (MbLiteral)
//        return FnNormal;
//    const auto& MnMin = (*MnMinPtr);
//    const auto& MnMax = (*MnMaxPtr);
//    return Normals::ToRaw(FnNormal, MnMin, MnMax);
//}


DXF void RA::Normals::Update()
{
    if (MvValues == nullptr || MnStorageSize == 0)
    {
        printf(RED "Normals needs storage to work\n" WHITE);
        return;
    }

    if (++MnRunningSize >= MnStorageSize)
        MnRunningSize = MnStorageSize;
    else
        return;

    auto LnIdx = *The.MnInsertIdxPtr + 1; // start at the beginnnig
    if (LnIdx >= MnStorageSize)
        LnIdx = 0;
    auto LnLoop = 0;
    do
    {
        if (MbLiteral)
            MvNormals[LnLoop] = MvValues[LnIdx];
        else
        {
            if (MeType == EType::Linear)
                MvNormals[LnLoop] = ToNormalLinear(MvValues[LnIdx]);
            else if (MeType == EType::Log)
                MvNormals[LnLoop] = ToRawLinear(MvValues[LnIdx]);
            else
                ThrowIt("Bad Code");
        }
        if (++LnIdx >= MnStorageSize)
            LnIdx = 0;
        ++LnLoop;
    } while (LnLoop < MnRunningSize); // you don't count yourself
    MnLastNormal = MvNormals[MnStorageSize - 1];
}

DXF void RA::Normals::Update(const double FnValue)
{
    if (MeType == EType::Linear)
        MnLastNormal = ToNormalLinear(FnValue);
    else if (MeType == EType::Log)
        MnLastNormal = ToRawLinear(FnValue);
    else
        ThrowIt("Bad Code");
}

DXF void RA::Normals::SetDefaultValues(const double FnDefaualt)
{
    MnLastNormal = 0;
}

RA::Normals::Config::Config(const xdbl FnCompression)
    : MnCompression(FnCompression)
{
}

RA::Normals::Config::Config(const xdbl FnCompression, const xdbl FnMin, const xdbl FnMax):
    MnCompression(FnCompression),
    MnMin(FnMin),
    MnMax(FnMax)
{
}

RA::Normals::Config::Config(const xdbl FnCompression, const xdbl* FnMinPtr, const xdbl* FnMaxPtr):
    MnCompression(FnCompression),
    MnMin(*FnMinPtr),
    MnMax(*FnMaxPtr)
{
}

RA::Normals::Config::Config(const xdbl FnCompression, const Config& FoConfig):
    MnCompression(FoConfig.MnCompression),
    MnMin(FoConfig.MnMin),
    MnMax(FoConfig.MnMax)
{
}

DXF xdbl RA::Normals::ToNormalLinear(const double& FnRaw)
{
    if (MbLiteral)
        return FnRaw;
    return ToNormalLinear(FnRaw, Config(MnCompression, MnMinPtr, MnMaxPtr));
}

DXF xdbl RA::Normals::ToRawLinear(const double& FnNormal)
{
    if (MbLiteral)
        return FnNormal;
    return ToNormalLinear(FnNormal, Config(MnCompression, MnMinPtr, MnMaxPtr));
}

DXF xdbl RA::Normals::ToNormalLog(const double& FnRaw)
{
    if (MbLiteral)
        return FnRaw;
    return ToNormalLog(FnRaw, Config(MnCompression, MnMinPtr, MnMaxPtr));
}

DXF xdbl RA::Normals::ToRawLog(const double& FnNormal)
{
    if (MbLiteral)
        return FnNormal;
    return ToRawLog(FnNormal, Config(MnCompression, MnMinPtr, MnMaxPtr));
}

DXF double RA::Normals::GetNormalOld(const xint Idx) const
{
    AssertDblRange(MvNormals.Size() - 1, Idx, 0);
    return MvNormals.First(Idx);
}

DXF double RA::Normals::GetNormalNew(const xint Idx) const
{
    AssertDblRange(MvNormals.Size() - 1, Idx, 0);
    return MvNormals.Last(Idx);
}
