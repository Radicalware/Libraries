#pragma once

#include "BasicCUDA.cuh"

namespace RA
{
    class Allocate
    {
    public:
        constexpr Allocate(const xint FnLength) noexcept : 
            MnLength(FnLength), MnObjSize(sizeof(xint)) {}

        constexpr Allocate(const xint FnLength, const xint FnObjSize) noexcept : 
            MnLength(FnLength), MnObjSize(FnObjSize) {}

        Allocate(const Allocate& Other) noexcept 
            { This = Other; }

        constexpr void operator=(const Allocate& Other) noexcept 
            { MnLength = Other.MnLength; MnObjSize = Other.MnObjSize; }

        constexpr xint GetLength() const { return MnLength; }
        constexpr xint GetUnitSize() const { return MnObjSize; }
        constexpr xint GetMallocSize() const { return MnLength * MnObjSize + sizeof(xint); }
        constexpr xint GetMemCopySize() const { return MnLength * MnObjSize; }

    private:
        xint MnLength;
        xint MnObjSize;
    };
};