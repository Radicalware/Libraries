#pragma once

#ifndef __CUDA_BRIDGE__
#define __CUDA_BRIDGE__

#include "CudaImport.cuh"
#include "Allocate.cuh"

#include "Iterator.h"
#include "Macros.h"
#include "SharedPtr.h"

#include<iostream>
#include<vector>
#include<sstream>
#include<math.h>
#include<string>
#include<algorithm>
#include<utility>

namespace RA
{
    template<typename T = int>
    class CudaBridge : public RA::Iterator<T>
    {
    public:
        ~CudaBridge();
        inline CudaBridge() {}

        CudaBridge(uint FnSize);
        CudaBridge(uint FnSize, uint FnMemSize);
        CudaBridge(const Allocate& FoAllocate);
        CudaBridge(SharedPtr<T*> FnArray, uint FnSize);
        CudaBridge(SharedPtr<T*> FnArray, uint FnSize, uint FnMemSize);
        CudaBridge(T* FnArray, uint FnSize);
        CudaBridge(T* FnArray, uint FnSize, uint FnMemSize);

        CudaBridge(const RA::CudaBridge<T>& Other);
        CudaBridge(RA::CudaBridge<T>&& Other) noexcept;

        CudaBridge(std::initializer_list<T> FnInit);

        void ResetIterator() { This.SetIterator(MvHost.Ptr(), &MnLeng); }

        void Initialize(uint FnSize);
        void Initialize(uint FnSize, uint FnMemSize);

        void operator=(const CudaBridge<T>& Other);
        void operator=(CudaBridge<T>&& Other) noexcept;

        void operator=(const std::vector<T>& Other);
        void operator=(std::vector<T>&& Other);

        T& operator[](const uint Idx) { return MvHost[Idx]; }
        const T& operator[](const uint Idx) const { return MvHost[Idx]; }

        RA::SharedPtr<T*> AllocateHost(const uint FnNewSize = 0);
        void AllocateDevice();

        void ZeroHostData();

        uint Size() const { return MnLeng; }
        uint GetUnitByteSize() const { return MnByteSize; }
        uint GetAllocationSize() const { return MnLeng * MnByteSize + sizeof(T); }
        Allocate GetAllocation() const { return Allocate(MnLeng, MnByteSize); }

        bool HasHostArray()   const { return bool(MvHost.get()); }
        bool HasDeviceArray() const { return MvDevice != nullptr; }

        void CopyHostToDevice();
        void CopyDeviceToHost();

        T* GetHost();
        const T* GetHost() const;

        T& GetHost(uint FnIdx);
        const T& GetHost(uint FnIdx) const;

        T* GetDevice();
        const T* GetDevice() const;
        T& GetDevice(uint FnIdx);
        const T& GetDevice(uint FnIdx) const;


        // =================================================================================================================
        // Reduction
        template<typename F>
        T ReductionCPU(F&& Function, uint Size = 0) const;
        template<typename F1, typename F2>
        T ReductionGPU(F1&& CudaFunction, F2&& HostFunction, const dim3& FnGrid, const dim3& FnBlock, const int ReductionSize = 0) const;
        // =================================================================================================================
        // Return Nothing from the GPU
        struct NONE
        {
            template<typename F, typename ...A>
            static void RunGPU(const dim3& FnGrid, const dim3& FnBlock, F&& Function, A&&... Args);

            template<typename F, typename ...A>
            static void RunCPU(F&& Function, A&&... Args);
        };
        // =================================================================================================================
        // Return an Array from the GPU
        struct ARRAY
        {
            template<typename F, typename ...A>
            static RA::CudaBridge<T> RunGPU(const Allocate& FoAllocate, const dim3& FnGrid, const dim3& FnBlock, F&& Function, A&&... Args);

            template<typename F, typename ...A>
            static RA::CudaBridge<T> RunCPU(const Allocate& FoAllocate, F&& Function, A&&... Args);
        };
        // =================================================================================================================
        // Misc
        static bool SameHostArrays(const RA::CudaBridge<T>& One, const RA::CudaBridge<T>& Two);
        static RA::CudaBridge<T> SumArraysIndicesCPU(const RA::CudaBridge<T>& One, const RA::CudaBridge<T>& Two);
        // -----------------------------------------------------------------------------------------------------------------

    private:
        void CopyHostData(const T* FnArray);
        void CopyHostData(const RA::SharedPtr<T*> FnArray);
        void CopyDeviceData(const T* FnArray);

        enum class MemType
        {
            IsHost,
            IsDevice
        };

        RA::SharedPtr<T*> MvHost = nullptr;
        T* MvDevice = nullptr;

        uint MnLeng = 0;
        uint MnByteSize = 0;
    };
};

template<typename T>
RA::CudaBridge<T>::~CudaBridge()
{
    if (MvDevice)
        cudaFree(MvDevice);// todo: add error check
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(uint FnSize) :
    MnLeng(FnSize), MnByteSize(sizeof(T))
{
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(uint FnSize, uint FnMemSize) :
    MnLeng(FnSize), MnByteSize(FnMemSize)
{
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(const RA::Allocate& FoAllocate):
    MnLeng(FoAllocate.GetLength()), MnByteSize(FoAllocate.GetUnitSize())
{
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(RA::SharedPtr<T*> FnArray, uint FnSize) :
    MvHost(FnArray), MnLeng(FnSize), MnByteSize(sizeof(T))
{
    SetIterator(MvHost.Ptr(), &MnLeng);
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(RA::SharedPtr<T*> FnArray, uint FnSize, uint FnMemSize) :
    MvHost(FnArray), MnLeng(FnSize), MnByteSize(FnMemSize)
{
    SetIterator(MvHost.Ptr(), &MnLeng);
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(T* FnArray, uint FnSize) :
    MvHost(RA::SharedPtr<T*>(FnArray)), MnLeng(FnSize), MnByteSize(sizeof(T))
{
    SetIterator(MvHost.Ptr(), &MnLeng);
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(T* FnArray, uint FnSize, uint FnMemSize) :
    MvHost(RA::SharedPtr<T*>(FnArray)), MnLeng(FnSize), MnByteSize(FnMemSize)
{
    SetIterator(MvHost.Ptr(), &MnLeng);
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(const RA::CudaBridge<T>& Other)
{
    *this = Other;
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(RA::CudaBridge<T>&& Other) noexcept
{
    *this = std::move(Other);
}

template<typename T>
RA::CudaBridge<T>::CudaBridge(std::initializer_list<T> FnInit)
{
    MnLeng = FnInit.size();
    if(!MnByteSize)
        MnByteSize = sizeof(T);
    AllocateHost();
    uint i = 0;
    for (const T& Val : FnInit)
    {
        MvHost[i] = Val;
        i++;
    }
}

template<typename T>
void RA::CudaBridge<T>::Initialize(uint FnSize)
{
    MvHost = nullptr;
    if (MvDevice)
        cudaFree(MvDevice);
    MvDevice = nullptr;

    if(FnSize) 
        MnLeng = FnSize;
    if (!MnByteSize) 
        MnByteSize = sizeof(T);
}

template<typename T>
void RA::CudaBridge<T>::Initialize(uint FnSize, uint FnMemSize)
{
    MvHost = nullptr;
    if (MvDevice)
        cudaFree(MvDevice);
    MvDevice = nullptr;

    MnLeng = FnSize;
    MnByteSize = FnMemSize;
}

template<typename T>
void RA::CudaBridge<T>::operator=(const RA::CudaBridge<T>& Other)
{
    MnLeng = Other.Size();
    MnByteSize = Other.MnByteSize;

    if (Other.HasHostArray())
    {
        AllocateHost();
        CopyHostData(Other.MvHost);
    }

    if (Other.HasDeviceArray())
    {
        AllocateDevice();
        CopyDeviceData(Other.MvDevice);
    }
    SetIterator(MvHost.Ptr(), &MnLeng);
}

template<typename T>
void RA::CudaBridge<T>::operator=(RA::CudaBridge<T>&& Other) noexcept
{
    MnLeng = Other.MnLeng;
    MnByteSize = Other.MnByteSize;

    if (Other.HasHostArray())
    {
        AllocateHost();
        MvHost = std::move(Other.MvHost);
    }
    SetIterator(MvHost.Ptr(), &MnLeng);
}

template<typename T>
void RA::CudaBridge<T>::operator=(const std::vector<T>& Other)
{
    if (!MnByteSize)
        throw "Cuda Bridge has 0 MnByteSize";

    MnLeng = Other.size();

    AllocateHost();
    CopyHostData(Other.data());
    SetIterator(MvHost.Ptr(), &MnLeng);
}

template<typename T>
void RA::CudaBridge<T>::operator=(std::vector<T>&& Other)
{
    if (!MnByteSize)
        throw "Cuda Bridge has 0 MnByteSize";

    MnLeng = Other.MnLeng;
    AllocateHost();
    for (uint i = 0; i < MnLeng; i++)
        std::swap_ranges(Other[i].begin(), Other[i].end(), &MvHost[i]);
    SetIterator(MvHost.Ptr(), &MnLeng);
}


template<typename T>
RA::SharedPtr<T*> RA::CudaBridge<T>::AllocateHost(const uint FnNewSize)
{
    Initialize(FnNewSize);
    MvHost = RA::MakeSharedBuffer<T>(MnLeng, MnByteSize);
    SetIterator(MvHost.Ptr(), &MnLeng);
    return MvHost;
}

template<typename T>
void RA::CudaBridge<T>::AllocateDevice()
{
    Begin();
    if (!MvHost)
        AllocateHost();

    if (MvDevice)
        cudaFree(MvDevice);

    auto FnError = cudaMalloc((T**)&MvDevice, GetAllocationSize());
    if (FnError)
        ThrowIt("Cuda Malloc Error: ", cudaGetErrorString(FnError));
    Rescue();
}

template<typename T>
void RA::CudaBridge<T>::ZeroHostData()
{
    std::fill(
        &MvHost[0], // fill data from here
        &MvHost[0] + MnLeng, // to here
        0); // with null data
}

template<typename T>
void RA::CudaBridge<T>::CopyHostToDevice()
{
    Begin();
    if(!MvDevice)
        AllocateDevice();
    auto Error = cudaMemcpy(MvDevice, MvHost.get(), GetAllocationSize(), cudaMemcpyHostToDevice);
    if (Error)
        ThrowIt("CUDA Memcpy Error: ", cudaGetErrorString(Error));
    SetIterator(MvHost.Ptr(), &MnLeng);
    Rescue();
}

template<typename T>
void RA::CudaBridge<T>::CopyDeviceToHost()
{
    Begin();
    if (!MvHost)
        AllocateHost();
    auto Error = cudaMemcpy(MvHost.get(), MvDevice, GetAllocationSize(), cudaMemcpyDeviceToHost);
    if (Error)
        ThrowIt("CUDA Memcpy Error: ", cudaGetErrorString(Error));
    Rescue();
}

template<typename T>
T* RA::CudaBridge<T>::GetHost()
{
    return MvHost.Raw();
}

template<typename T>
const T* RA::CudaBridge<T>::GetHost() const
{
    return MvHost.Raw();
}

template<typename T>
inline T& RA::CudaBridge<T>::GetHost(uint FnIdx)
{
    if (FnIdx >= MnLeng)
        throw "RA::CudaBridge<T>::Get >> FnIdx Is too big";

    return MvHost[FnIdx];
}

template<typename T>
const T& RA::CudaBridge<T>::GetHost(uint FnIdx) const
{
    if (FnIdx >= MnLeng)
        throw "RA::CudaBridge<T>::Get >> FnIdx Is too big";

    return MvHost[FnIdx];
}

template<typename T>
T* RA::CudaBridge<T>::GetDevice()
{
    return MvDevice;
}

template<typename T>
const T* RA::CudaBridge<T>::GetDevice() const
{
    return MvDevice;
}

template<typename T>
inline T& RA::CudaBridge<T>::GetDevice(uint FnIdx)
{
    if (FnIdx >= MnLeng)
        throw "RA::CudaBridge<T>::Get >> FnIdx Is too big";

    return MvDevice[FnIdx];
}

template<typename T>
const T& RA::CudaBridge<T>::GetDevice(uint FnIdx) const
{
    if (FnIdx >= MnLeng)
        throw "RA::CudaBridge<T>::Get >> FnIdx Is too big";

    return MvDevice[FnIdx];
}

template<typename T>
template<typename F>
T RA::CudaBridge<T>::ReductionCPU(F&& Function, uint Size) const
{
    //RA::CudaBridge<T> HostOutput(MnLeng);
    //HostOutput.AllocateHost();
    //HostOutput.ZeroHostData();
    //HostOutput.AllocateDevice();
    //HostOutput.CopyHostToDevice();
    if (Size == 0)
        return Function(MvHost.get(), MnLeng);
    return Function(MvHost.get(), Size);
}

template<typename T>
template<typename F1, typename F2>
T RA::CudaBridge<T>::ReductionGPU(F1&& CudaFunction, F2&& HostFunction, const dim3& FnGrid, const dim3& FnBlock, const int ReductionSize) const
{
    RA::CudaBridge<T> Data = *this;
    Data.AllocateDevice();
    Data.CopyHostToDevice();
    RA::CudaBridge<T> ReturnedArray(Size());
    ReturnedArray.AllocateDevice();
    CudaFunction<<<FnGrid, FnBlock>>>(Data.GetDevice(), ReturnedArray.GetDevice(), static_cast<int>(Size()));
    cudaDeviceSynchronize();
    ReturnedArray.CopyDeviceToHost();
    if (ReductionSize == 0)
        return ReturnedArray.ReductionCPU(HostFunction, Size());
    return ReturnedArray.ReductionCPU(HostFunction, ReductionSize);
}

template<typename T>
std::ostream& operator<<(std::ostream& out, RA::CudaBridge<T>& obj)
{
    out << obj.GetStr();
    return out;
}

template<typename T>
template<typename F, typename ...A>
void RA::CudaBridge<T>::NONE::RunGPU(const dim3& FnGrid, const dim3& FnBlock, F&& Function, A&&... Args)
{
    Begin();
    Function<<<FnGrid, FnBlock>>>(std::forward<A>(Args)...);
    cudaDeviceSynchronize();
    Rescue();
}

template<typename T>
template<typename F, typename ...A>
void RA::CudaBridge<T>::NONE::RunCPU(F&& Function, A&&... Args)
{
    Begin();
    Function(std::forward<A>(Args)...);
    Rescue();
}


template<typename T>
template<typename F, typename ...A>
RA::CudaBridge<T> RA::CudaBridge<T>::ARRAY::RunGPU(const RA::Allocate& FoAllocate, const dim3& FnGrid, const dim3& FnBlock, F&& Function, A&&... Args)
{
    Begin();
    RA::CudaBridge<T> DeviceOutput(FoAllocate);
    DeviceOutput.AllocateHost();
    DeviceOutput.AllocateDevice();
    DeviceOutput.CopyHostToDevice();
    
    Function<<<FnGrid, FnBlock>>>(DeviceOutput.GetDevice(), std::forward<A>(Args)...);
    cudaDeviceSynchronize();

    DeviceOutput.CopyDeviceToHost();

    return DeviceOutput;
    Rescue();
}


template<typename T>
template<typename F, typename ...A>
RA::CudaBridge<T> RA::CudaBridge<T>::ARRAY::RunCPU(const RA::Allocate& FoAllocate, F&& Function, A&&... Args)
{
    Begin();
    RA::CudaBridge<T> HostOutput(FoAllocate);
    HostOutput.AllocateHost();
    Function(HostOutput.GetHost(), std::forward<A>(Args)...);
    return HostOutput;
    Rescue();
}

template<typename T>
bool RA::CudaBridge<T>::SameHostArrays(const RA::CudaBridge<T>& One, const RA::CudaBridge<T>& Two)
{
    for (uint i = 0; i < One.Size(); i++)
    {
        if (One.MvHost[i] != Two.MvHost[i])
            return false;
    }
    return true;
}

template<typename T>
RA::CudaBridge<T> RA::CudaBridge<T>::SumArraysIndicesCPU(
    const RA::CudaBridge<T>& One, const RA::CudaBridge<T>& Two)
{
    RA::CudaBridge<T> LoCombinedArray(One.Size());
    LoCombinedArray.AllocateHost();
    LoCombinedArray.ZeroHostData();

    for (uint i = 0; i < One.Size(); i++)
        LoCombinedArray.MvHost[i] += One.MvHost[i] + Two.MvHost[i];
    return LoCombinedArray;
}

template<typename T>
void RA::CudaBridge<T>::CopyHostData(const T* FnArray)
{
    AllocateHost();
    for (uint i = 0; i < MnLeng; i++)
        MvHost[i] = FnArray[i];
}

template<typename T>
void RA::CudaBridge<T>::CopyHostData(const RA::SharedPtr<T*> FnArray)
{
    AllocateHost();
    for (uint i = 0; i < MnLeng; i++)
        MvHost[i] = FnArray[i];
}

template<typename T>
void RA::CudaBridge<T>::CopyDeviceData(const T* FnArray)
{
    AllocateDevice();
    if (!FnArray)
        return;
    for (uint i = 0; i < MnLeng; i++)
        MvDevice[i] = FnArray[i];
}

#endif