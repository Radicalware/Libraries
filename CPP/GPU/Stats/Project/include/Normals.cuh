#pragma once

#ifndef UsingMSVC
#include "ImportCUDA.cuh"
#else
#include "xvector.h"
#endif
#include "RawMapping.h"

namespace RA
{
    class Normals
    {
        friend class Stats;
        constexpr istatic dbl SnUpper = 2.0;
        constexpr istatic dbl SnLower = 1.0;

    public:
        enum class EType
        {
            Linear,
            Log
        };
        struct Config
        {
            DXF Config(const dbl FnCompression);
            DXF Config(const dbl FnCompression, const dbl  FnMin, const dbl  FnMax);
            DXF Config(const dbl FnCompression, const dbl* FnMinPtr, const dbl* FnMaxPtr);
            DXF Config(const dbl FnCompression, const Config& FoConfig);
            const dbl MnCompression = 1;
            const dbl MnMax = -DBL_MAX;
            const dbl MnMin = DBL_MAX;
        };

        DXF void CopyStats(const Normals& Other);
        Normals(
            const EHardware FeHardware,
            const double* FvValues,
            const   xint* FnInsertIdxPtr,
            const   dbl* FnMinPtr,
            const   dbl* FnMaxPtr,
            const   xint  FnStorageSize = 0);

        ~Normals();

        IXF void SetCompression(const double FnCompress) { MnCompression = FnCompress; }
        IXF auto SetType(EType FeType) { MeType = FeType; }

        IXF auto BxLiteral()      const { return MbLiteral; }
        IXF auto GetCompression() const { return MnCompression; }
        IXF auto GetMax()         const { return *MnMaxPtr; }
        IXF auto GetCurrent()     const { return (MvValues) ? MvValues[*MnInsertIdxPtr] : MnLastNormal; }
        IXF auto GetMin()         const { return *MnMinPtr; }
        IXF auto GetLastNormal()  const { return MnLastNormal; }
        IXF auto BxFullSize()     const { return MnStorageSize == MnRunningSize; }
        IXF auto GetRunningSize() const { return MnRunningSize; }


        IXF static constexpr dbl ToNormalLinear(const double& FnRaw, const Config& FoConfig);
        IXF static constexpr dbl ToRawLinear(const double& FnCompressedNormal, const Config& FoConfig);

        IXF static constexpr dbl ToNormalLog(const double& FnRaw, const Config& FoConfig);
        IXF static constexpr dbl ToRawLog(const double& FnCompressedNormal, const Config& FoConfig);

        DXF dbl ToNormalLinear(const double& FnRaw);
        DXF dbl ToRawLinear(const double& FnNormal);

        DXF dbl ToNormalLog(const double& FnRaw);
        DXF dbl ToRawLog(const double& FnNormal);

        IXF void SetLiteral(const bool Truth) { MbLiteral = Truth; }

#ifdef UsingMSVC
        DXF const xvector<double>& GetNormals() const { return MvNormals; }
        DXF double GetNormalFront(const xint Idx) const;
        DXF double GetNormalBack(const xint Idx) const;
#else // UsingMSVC
        DXF const double* GetNormals() const { return MvNormals; }
        DXF double GetNormal(const xint Idx) const { return MvNormals[Idx]; }
        DXF double GetNormalFront(const xint Idx) const;
        DXF double GetNormalBack(const xint Idx) const;

#endif
    private:
        DXF void Update();
        DXF void Update(const double FnValue);
        DXF void SetDefaultValues(const double FnDefaualt);

        EHardware MeHardware = EHardware::Default;
        EType MeType = EType::Linear;
        const double* MvValues = nullptr; // end point slides with MnInsertIdxPtr
        const dbl* MnMinPtr = nullptr;
        const dbl* MnMaxPtr = nullptr;
        const xint* MnInsertIdxPtr;
        xint        MnStorageSize;
        bool        MbLiteral = false;
        dbl        MnCompression = 1.0;

        double      MnLastNormal = 0;
        xint        MnRunningSize = 0;

#ifdef UsingMSVC
        xvector<double> MvNormals;
#else
        double* MvNormals = nullptr; // order old to new >> 0 to (size - 1)
#endif
    };
}



IXF constexpr dbl RA::Normals::ToNormalLinear(const double& FnRaw, const Config& FoConfig)
{
    if (Appx(FoConfig.MnMin, FoConfig.MnMax))
        return 0; // 0 is the centerpoint so min/max even means you are at zero

    const auto LnNormal = 2.0 * ((FnRaw - FoConfig.MnMin) / (FoConfig.MnMax - FoConfig.MnMin)) - 1.0;
    const auto LnCompressedNormal = LnNormal / FoConfig.MnCompression;
    return LnCompressedNormal;
}

IXF constexpr dbl RA::Normals::ToRawLinear(const double& FnNormal, const Config& FoConfig)
{
    if (Appx(FoConfig.MnMin, FoConfig.MnMax))
        return FoConfig.MnMax;

    const auto LoUncompressedNormal = FnNormal * FoConfig.MnCompression;
    return ((LoUncompressedNormal + 1.0) / 2.0) * (FoConfig.MnMax - FoConfig.MnMin) + FoConfig.MnMin;
}


IXF constexpr dbl RA::Normals::ToNormalLog(const double& FnRaw, const Config& FoConfig)
{
    if (Appx(FoConfig.MnMin, FoConfig.MnMax))
        return 0;

    cvar LnMidpoint = (FoConfig.MnMin + FoConfig.MnMax) / 2.0;
    if (FnRaw > LnMidpoint)
    {
        cvar LnLinearNormal = ToNormalLinear(FnRaw, Config(1, FoConfig));
        cvar LnLogNormal(log(LnLinearNormal + 1.0) / log(2.0));
        cvar LnLogCompressed = LnLogNormal / FoConfig.MnCompression;
        return LnLogCompressed;
    }
    else if (FnRaw < LnMidpoint)
    {
        // 15 8 5
        // midpoint 10
        // 10 - 8 = 2
        // (2 * 2) + 8 = 12
        cvar LnRaw = ((LnMidpoint - FnRaw) * 2) + FnRaw;
        cvar FnNormalized = ToNormalLinear(LnRaw, Config(1, FoConfig));
        cvar LnLogNormal = -(log(FnNormalized + 1.0) / log(2.0));
        cvar LnLogCompressed = LnLogNormal / FoConfig.MnCompression;
        return LnLogCompressed;
    }
    return 0;
}

IXF constexpr dbl RA::Normals::ToRawLog(const double& FnCompressedNormal, const Config& FoConfig)
{
    if (Appx(FoConfig.MnMin, FoConfig.MnMax))
        return FoConfig.MnMax;

    //cvar FnUncompressed = FnCompressedNormal * FoConfig.MnCompression;
    dbl LoLogUncompressed = 0.0L;
    if (FnCompressedNormal > 0)
        LoLogUncompressed = exp(FnCompressedNormal * log(2.0)) - 1.0;
    else if (FnCompressedNormal < 0)
        LoLogUncompressed = -(exp(-FnCompressedNormal * log(2.0)) - 1.0);
    else
        LoLogUncompressed = 0.0L;

    cvar LnRaw = ToRawLinear(LoLogUncompressed, Config(1, FoConfig));
    return LnRaw;
}


