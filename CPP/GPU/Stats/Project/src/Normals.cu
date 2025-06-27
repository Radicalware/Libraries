
// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]
#if UsingMSVC
#include "Normals.h"
#else
#include "Normals.cuh"
#endif
#include "xstring.h"
#include "Macros.h"

DXF void RA::Normals::CopyStats(const RA::Normals& Other)
{
    MeHardware = Other.MeHardware;
    MeType = Other.MeType;
    MvValues = Other.MvValues;
    MnMinPtr = Other.MnMinPtr;
    MnMaxPtr = Other.MnMaxPtr;
    MnInsertIdxPtr = Other.MnInsertIdxPtr;
    MnStorageSize = Other.MnStorageSize;
    MbLiteral = Other.MbLiteral;
    MnCompression = Other.MnCompression;

    MnLastNormal = Other.MnLastNormal;
    MnRunningSize = Other.MnRunningSize;
}

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
    return ToRawLinear(FnNormal, Config(MnCompression, MnMinPtr, MnMaxPtr));
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

DXF double RA::Normals::GetNormalFront(const xint Idx) const
{
    AssertDblRange(MvNormals.Size() - 1, Idx, 0);
    return MvNormals.First(Idx);
}

DXF double RA::Normals::GetNormalBack(const xint Idx) const
{
    AssertDblRange(MvNormals.Size() - 1, Idx, 0);
    return MvNormals.Last(Idx);
}
