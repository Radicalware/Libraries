#pragma once

#include "CudaImport.cuh"

namespace RA
{
    class Allocate
    {
    public:
        constexpr Allocate(const uint FnLength) noexcept : 
            MnLength(FnLength), MnObjSize(sizeof(uint)) {}

        constexpr Allocate(const uint FnLength, const uint FnObjSize) noexcept : 
            MnLength(FnLength), MnObjSize(FnObjSize) {}

        Allocate(const Allocate& Other) noexcept 
            { This = Other; }

        constexpr void operator=(const Allocate& Other) noexcept 
            { MnLength = Other.MnLength; MnObjSize = Other.MnObjSize; }

        constexpr uint GetLength() const { return MnLength; }
        constexpr uint GetUnitSize() const { return MnObjSize; }
        constexpr uint GetAllocationSize() const { return MnLength * MnObjSize + sizeof(uint); }
        constexpr uint GetDataByteSize() const { return MnLength * MnObjSize; }

    private:
        uint MnLength;
        uint MnObjSize;
    };
};