#pragma once

/*
*|| Copyright[2023][Joel Leagues aka Scourge]
*|| Scourge /at\ protonmail /dot\ com
*|| www.Radicalware.net
*|| https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*||
*|| Licensed under the Apache License, Version 2.0 (the "License");
*|| you may not use this file except in compliance with the License.
*|| You may obtain a copy of the License at
*||
*|| http ://www.apache.org/licenses/LICENSE-2.0
*||
*|| Unless required by applicable law or agreed to in writing, software
*|| distributed under the License is distributed on an "AS IS" BASIS,
*|| WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*|| See the License for the specific language governing permissions and
*|| limitations under the License.
*/

#ifndef __CUDA_BRIDGE__
#define __CUDA_BRIDGE__

#include "ImportCUDA.cuh"
#include "Allocate.cuh"
#include "Host.cuh"

#include "Iterator.h"
#include "Nexus.h"
#include "Macros.h"
#include "SharedPtr/SharedPtrArr.h"
#include "xvector.h"

namespace RA
{
    template<typename T = int>
    class CudaBridge : public RA::Iterator<T>
    {
    public:
        INL static void SetDeviceIdx(const int Idx);
        INL static auto GetDeviceIdx() { return SnDeviceIdx.load(); }
        INL static int  GetDeviceCount();

        INL void FreeDevice();
        INL ~CudaBridge();

        INL static void Initialize();
        INL CudaBridge() {}

        INL CudaBridge(uint FnSize);
        INL CudaBridge(uint FnSize, uint FnMemSize);
        INL CudaBridge(const Allocate& FoAllocate);
        INL CudaBridge(SharedPtr<T[]> FnArray, uint FnSize);
        INL CudaBridge(SharedPtr<T[]> FnArray, uint FnSize, uint FnMemSize);

        INL CudaBridge(const RA::CudaBridge<T>& Other);
        INL CudaBridge(RA::CudaBridge<T>&& Other) noexcept;

        INL CudaBridge(std::initializer_list<T> FnInit);

        INL void ResetIterator() { This.SetIterator(MvHost.Ptr(), &MnLeng); }

        INL void SetSizing(uint FnSize);
        INL void SetSizing(uint FnSize, uint FnMemSize);

        INL void operator=(const CudaBridge<T>& Other);
        INL void operator=(CudaBridge<T>&& Other) noexcept;

        INL void operator=(const std::vector<T>& Other);
        INL void operator=(std::vector<T>&& Other);

        INL T& operator[](const uint Idx) { return MvHost[Idx]; }
        INL const T& operator[](const uint Idx) const { return MvHost[Idx]; }

        template<typename ...A>
        INL void AllocateHost(const uint FnNewSize = 0, A&&... Args);
        INL void AllocateDevice();

        INL void ZeroHostData();

        INL uint Size() const { return MnLeng; }
        INL uint GetUnitByteSize() const { return MnByteSize; }
        INL uint GetMallocSize()   const { return MnLeng * MnByteSize + sizeof(uint); }
        INL uint GetMemCopySize()  const { return MnLeng * MnByteSize; }
        INL Allocate GetAllocation() const { return Allocate(MnLeng, MnByteSize); }

        INL bool HasHostArray()   const { return bool(MvHost.Ptr()); }
        INL bool HasDeviceArray() const { return MvDevice != nullptr; }

        INL void CopyHostToDevice();
        INL void CopyDeviceToHost();

        INL void CopyHostToDeviceAsync();
        INL void CopyDeviceToHostAsync();
        INL void SyncStream() const;
        
        INL       T& Get()              { return MvHost[0]; }
        INL const T& Get()        const { return MvHost[0]; }
        INL       T& GetHostObj()       { return MvHost[0]; }
        INL const T& GetHostObj() const { return MvHost[0]; }

        INL const auto& GetStream() const { return MoStream; }
        INL       T*    GetHost();
        INL const T*    GetHost() const;
        INL xvector<T>  GetVec()  const;

        INL       SharedPtr<T[]>& GetShared();
        INL const SharedPtr<T[]>& GetShared() const;

        INL       T& GetHost(uint FnIdx);
        INL const T& GetHost(uint FnIdx) const;

        INL       T* GetDevice();
        INL const T* GetDevice() const;


        // =================================================================================================================
        // Reduction
        template<typename F>
        INL T ReductionCPU(F&& Function, uint Size = 0) const;
        template<typename F1, typename F2>
        INL T ReductionGPU(F1&& CudaFunction, F2&& HostFunction, const dim3& FnGrid, const dim3& FnBlock, const int ReductionSize = 0) const;
        // =================================================================================================================
        // Return Nothing from the GPU
        struct NONE
        {
            template<typename F, typename ...A>
            INL static void RunGPU(const dim3& FnGrid, const dim3& FnBlock, F&& Function, A&&... Args);

            template<typename F, typename ...A>
            INL static void RunCPU(F&& Function, A&&... Args);

            template<typename F, typename ...A>
            INL static xvector<T>
                RunMultiGPU(
                    const uint FnSize,
                    F&& Function, A&&... Args); // CUDA Kernel Function & Args
        };
        // =================================================================================================================
        // Return an Array from the GPU
        struct ARRAY
        {
            template<typename F, typename ...A>
            INL static CudaBridge<T> RunGPU(const Allocate& FoAllocate, const dim3& FnGrid, const dim3& FnBlock, F&& Function, A&&... Args);
            template<typename F, typename ...A>
            INL static xvector<xp<CudaBridge<T>>>
                RunUnjoinedMultiGPU(
                    const Allocate& FoAllocate, // Return config
                    const dim3& FnGrid, const dim3& FnBlock, // CUDA Kernel Config
                    F&& Function, A&&... Args); // CUDA Kernel Function & Args
            template<typename F, typename ...A>
            INL static xvector<T>
                RunMultiGPU(
                    const Allocate& FoAllocate, // Return config
                    F&& Function, A&&... Args); // CUDA Kernel Function & Args


            template<typename F, typename ...A>
            INL static CudaBridge<T> RunCPU(const Allocate& FoAllocate, F&& Function, A&&... Args);
        };


        inline static std::vector<cudaStream_t> SvStreams;
        static void SyncAll();

        // =================================================================================================================
        // Misc
        INL static bool SameArrays(const RA::CudaBridge<T>& One, const RA::CudaBridge<T>& Two);
        INL static bool SameArrays(const xvector<T>& One, const xvector<T>& Two);
        INL static CudaBridge<T> SumArraysIndicesCPU(const RA::CudaBridge<T>& One, const RA::CudaBridge<T>& Two);
        // -----------------------------------------------------------------------------------------------------------------

    private:
        INL void CopyHostData(const T* FnArray);
        INL void CopyHostData(const RA::SharedPtr<T[]>& FnArray);

        enum class MemType
        {
            IsHost,
            IsDevice
        };

        SharedPtr<T[]> MvHost = nullptr;
        T* MvDevice = nullptr;

        uint MnLeng = 0;
        uint MnByteSize = 0;

        cudaStream_t MoStream = 0;
        bool MbDelete = true;

        INL static Atomic<int>  SnDeviceIdx = 0;
        INL static Atomic<int>  SnDeviceCount = 0;
        INL static Atomic<bool> SbInitialized = false;
    };
};

TTT inline void RA::CudaBridge<T>::SetDeviceIdx(const int Idx)
{
    Begin();
    if (SnDeviceIdx == Idx)
        return;
    SnDeviceIdx = Idx;
    auto Error = cudaSetDevice(Idx);
    if (Error)
        ThrowIt("CUDA Set Device (", Idx, ") Error: ", cudaGetErrorString(Error));
    Rescue();
}

TTT inline int RA::CudaBridge<T>::GetDeviceCount()
{
    Begin();
    if (SnDeviceCount != 0)
        return SnDeviceCount;
    Initialize();
    return SnDeviceCount;
    Rescue();
}

TTT void RA::CudaBridge<T>::FreeDevice()
{
    if (!MbDelete)
        return;
    if (MvDevice)
    {
        cudaFree(MvDevice);
        MvDevice = nullptr;
    }
    if (MoStream)
        cudaStreamDestroy(MoStream);
}

TTT void RA::CudaBridge<T>::Initialize()
{
    Begin();
    if (SnDeviceCount == 0)
    {
        int LnDeviceCount = 0;
        cudaGetDeviceCount(&LnDeviceCount);
        SnDeviceCount = LnDeviceCount;
    }

    if (!SbInitialized)
    {
        for (int i = 0; i < SnDeviceCount; i++)
        {
            for (int j = 0; j < SnDeviceCount; j++)
            {
                if (i == j)
                    continue;
                int LbTruth = 0;
                auto Error = cudaDeviceCanAccessPeer(&LbTruth, i, j);
                if (Error)
                    ThrowIt("CUDA cudaDeviceCanAccessPeer Error: ", cudaGetErrorString(Error));
                if (LbTruth == false)
                    ThrowIt("You have no P2P access on your current devices!");
            }
        }

        // Enable P2P Access
        for (int i = 0; i < SnDeviceCount; i++)
        {
            for (int j = 0; j < SnDeviceCount; j++)
            {
                if (i == j)
                    continue;
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
        SbInitialized = true;
    }
    Rescue();
}

TTT RA::CudaBridge<T>::~CudaBridge()
{
    FreeDevice();
}

TTT RA::CudaBridge<T>::CudaBridge(uint FnSize) :
    MnLeng(FnSize), MnByteSize(sizeof(T))
{
    Initialize();
}

TTT RA::CudaBridge<T>::CudaBridge(uint FnSize, uint FnMemSize) :
    MnLeng(FnSize), MnByteSize(FnMemSize)
{
    Initialize();
}

TTT RA::CudaBridge<T>::CudaBridge(const RA::Allocate& FoAllocate):
    MnLeng(FoAllocate.GetLength()), MnByteSize(FoAllocate.GetUnitSize())
{
    Initialize();
}

TTT RA::CudaBridge<T>::CudaBridge(RA::SharedPtr<T[]> FnArray, uint FnSize) :
    MvHost(FnArray), MnLeng(FnSize), MnByteSize(sizeof(T))
{
    Initialize();
    SetIterator(MvHost.Ptr(), &MnLeng);
}

TTT RA::CudaBridge<T>::CudaBridge(RA::SharedPtr<T[]> FnArray, uint FnSize, uint FnMemSize) :
    MvHost(FnArray), MnLeng(FnSize), MnByteSize(FnMemSize)
{
    Initialize();
    SetIterator(MvHost.Ptr(), &MnLeng);
}

TTT RA::CudaBridge<T>::CudaBridge(const RA::CudaBridge<T>& Other)
{
    *this = Other;
}

TTT RA::CudaBridge<T>::CudaBridge(RA::CudaBridge<T>&& Other) noexcept
{
    *this = std::move(Other);
}

TTT RA::CudaBridge<T>::CudaBridge(std::initializer_list<T> FnInit)
{
    Begin();
    Initialize();
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
    Rescue();
}

TTT void RA::CudaBridge<T>::SetSizing(uint FnSize)
{
    if(FnSize) 
        MnLeng = FnSize;
    if (!MnByteSize) 
        MnByteSize = sizeof(T);
}

TTT void RA::CudaBridge<T>::SetSizing(uint FnSize, uint FnMemSize)
{
    MnLeng = FnSize;
    MnByteSize = FnMemSize;
}

TTT void RA::CudaBridge<T>::operator=(const RA::CudaBridge<T>& Other)
{
    Begin();
    MnByteSize = Other.MnByteSize;
    MoStream = Other.MoStream;
    if (Other.HasHostArray())
    {
        AllocateHost(Other.Size());
        CopyHostData(Other.MvHost);

        AllocateDevice();
        CopyHostToDevice();
    }
    MnLeng = Other.Size();

    This.SetIterator(MvHost.Ptr(), &MnLeng);
    Rescue();
}

TTT void RA::CudaBridge<T>::operator=(RA::CudaBridge<T>&& Other) noexcept
{
    Other.MbDelete = false;
    MnByteSize = Other.MnByteSize;
    MoStream = Other.MoStream;
    if (Other.HasHostArray())
        MvHost = std::move(Other.MvHost);
    MvDevice = Other.MvDevice;
    MnLeng = Other.MnLeng;
    This.SetIterator(MvHost.Ptr(), &MnLeng);
}

TTT void RA::CudaBridge<T>::operator=(const std::vector<T>& Other)
{
    Begin();
    if (!MnByteSize)
        throw "Cuda Bridge has 0 MnByteSize";

    MoStream = Other.MoStream;
    AllocateHost(Other.size());
    CopyHostData(Other.data());
    SetIterator(MvHost.Ptr(), &MnLeng);
    Rescue();
}

TTT void RA::CudaBridge<T>::operator=(std::vector<T>&& Other)
{
    Begin();
    if (!MnByteSize)
        throw "Cuda Bridge has 0 MnByteSize";

    MoStream = Other.MoStream;
    AllocateHost(Other.size());
    for (uint i = 0; i < MnLeng; i++)
        std::swap_ranges(Other.begin(), Other.end(), &MvHost[i]);
    SetIterator(MvHost.Ptr(), &MnLeng);
    Rescue();
}

TTT template<typename ...A>
void RA::CudaBridge<T>::AllocateHost(const uint FnNewSize, A&&... Args)
{
    Begin();
    if (MvHost.Size() != FnNewSize || !MvHost) // no reason to re-allocate if you already have the buffer
    {
        if ((FnNewSize == MnLeng || FnNewSize == 0) && !!MvHost) // 0 will be current MnLeng AND don't recreate exact same MvHost
            return;
        SetSizing(FnNewSize);
        MvHost = RA::SharedPtr<T[]>(MnLeng, std::forward<A>(Args)...);
        This.SetIterator(MvHost.Ptr(), &MnLeng);
        if (MvDevice)
            FreeDevice();
    }
    Rescue();
}

TTT void RA::CudaBridge<T>::AllocateDevice()
{
    Begin();
    if (!MvHost)
        AllocateHost();

    if (MvDevice)
        FreeDevice();

    auto FnError = cudaMalloc((T**)&MvDevice, GetMallocSize());
    if (FnError)
        ThrowIt("Cuda Malloc Error: ", cudaGetErrorString(FnError));
    Rescue();
}

TTT void RA::CudaBridge<T>::ZeroHostData()
{
    std::fill(
        &MvHost[0], // fill data from here
        &MvHost[0] + MnLeng, // to here
        0); // with null data
}

TTT void RA::CudaBridge<T>::CopyHostToDevice()
{
    Begin();
    if (!MvDevice)
        AllocateDevice();
    auto Error = cudaMemcpy(MvDevice, MvHost.Ptr(), GetMemCopySize(), cudaMemcpyHostToDevice);
    if (Error)
        ThrowIt("CUDA Memcpy Error Host>>Device: ", cudaGetErrorString(Error));
    Rescue();
}

TTT void RA::CudaBridge<T>::CopyDeviceToHost()
{
    Begin();
    if (!MvHost)
        AllocateHost();

    if (MoStream)
    {
        This.SyncStream();
        auto Error = cudaMemcpyAsync(MvHost.Ptr(), MvDevice, GetMemCopySize(), cudaMemcpyDeviceToHost, MoStream);
        if (Error)
            ThrowIt("CUDA cudaMemcpyAsync Error Host>>Device: ", cudaGetErrorString(Error));
    }
    else
    {
        auto Error = cudaMemcpy(MvHost.Ptr(), MvDevice, GetMemCopySize(), cudaMemcpyDeviceToHost);
        if (Error)
            ThrowIt("CUDA cudaMemcpy Error Host>>Device: ", cudaGetErrorString(Error));
    }
    This.SetIterator(MvHost.Ptr(), &MnLeng);
    Rescue();
}

TTT void RA::CudaBridge<T>::CopyHostToDeviceAsync()
{
    Begin();
    if (!MvDevice)
        AllocateDevice();

    if (MoStream == 0)
        cudaStreamCreateWithFlags(&MoStream, cudaStreamNonBlocking);
    auto Error = cudaMemcpyAsync(MvDevice, MvHost.Ptr(), GetMemCopySize(), cudaMemcpyHostToDevice, MoStream);
    if (Error)
        ThrowIt("CUDA Memcpy Error Host>>Device: ", cudaGetErrorString(Error));
    Rescue();
}

TTT void RA::CudaBridge<T>::CopyDeviceToHostAsync()
{
    Begin();
    if (!MvHost)
        AllocateHost();

    if (MoStream)
    {
        auto Error = cudaMemcpyAsync(MvHost.Ptr(), MvDevice, GetMemCopySize(), cudaMemcpyDeviceToHost, MoStream);
        if (Error)
            ThrowIt("CUDA Memcpy Error Device>>Host: ", cudaGetErrorString(Error));
    }
    else
    {
        auto Error = cudaMemcpyAsync(MvHost.Ptr(), MvDevice, GetMemCopySize(), cudaMemcpyDeviceToHost);
        if (Error)
            ThrowIt("CUDA Memcpy Error Device>>Host: ", cudaGetErrorString(Error));
    }


    This.SetIterator(MvHost.Ptr(), &MnLeng);
    Rescue();
}

TTT void RA::CudaBridge<T>::SyncStream() const
{
    if (MoStream)
    {
        auto Error = cudaStreamSynchronize(MoStream);
        if (Error)
            ThrowIt("CUDA Kernel Sync Stream Error on GPU #", SnDeviceIdx, ": ", cudaGetErrorString(Error));
    }
}

TTT T* RA::CudaBridge<T>::GetHost()
{
    return MvHost.Raw();
}

TTT const T* RA::CudaBridge<T>::GetHost() const
{
    return MvHost.Raw();
}

TTT xvector<T> RA::CudaBridge<T>::GetVec() const
{
    xvector<T> LvHost;
    if (!MnLeng || !MnByteSize)
        return LvHost;
    for (auto& Val : This)
        LvHost << Val;
    return LvHost;
}

TTT RA::SharedPtr<T[]>& RA::CudaBridge<T>::GetShared()
{
    return MvHost;
}

TTT const RA::SharedPtr<T[]>& RA::CudaBridge<T>::GetShared() const
{
    return MvHost;
}

TTT inline T& RA::CudaBridge<T>::GetHost(uint FnIdx)
{
    if (FnIdx >= MnLeng)
        throw "RA::CudaBridge<T>::Get >> FnIdx Is too big";

    return MvHost[FnIdx];
}

TTT const T& RA::CudaBridge<T>::GetHost(uint FnIdx) const
{
    if (FnIdx >= MnLeng)
        throw "RA::CudaBridge<T>::Get >> FnIdx Is too big";

    return MvHost[FnIdx];
}

TTT T* RA::CudaBridge<T>::GetDevice()
{
    return MvDevice;
}

TTT const T* RA::CudaBridge<T>::GetDevice() const
{
    return MvDevice;
}

TTT template<typename F>
T RA::CudaBridge<T>::ReductionCPU(F&& Function, uint Size) const
{
    //RA::CudaBridge<T> HostOutput(MnLeng);
    //HostOutput.AllocateHost();
    //HostOutput.ZeroHostData();
    //HostOutput.AllocateDevice();
    //HostOutput.CopyHostToDevice();
    if (Size == 0)
        return Function(MvHost.Ptr(), MnLeng);
    return Function(MvHost.Ptr(), Size);
}

TTT template<typename F1, typename F2>
T RA::CudaBridge<T>::ReductionGPU(F1&& CudaFunction, F2&& HostFunction, const dim3& FnGrid, const dim3& FnBlock, const int ReductionSize) const
{
    Begin();
    RA::CudaBridge<T> Data = *this;
    Data.AllocateDevice();
    Data.CopyHostToDevice();
    RA::CudaBridge<T> ReturnedArray(Size());
    ReturnedArray.AllocateDevice();
    CudaFunction<<<FnGrid, FnBlock>>>(Data.GetDevice(), ReturnedArray.GetDevice(), static_cast<int>(Size()));
    auto Error = cudaDeviceSynchronize();
    if (Error)
        ThrowIt("CUDA Kernel: ", cudaGetErrorString(Error));    ReturnedArray.CopyDeviceToHost();
    if (ReductionSize == 0)
        return ReturnedArray.ReductionCPU(HostFunction, Size());
    return ReturnedArray.ReductionCPU(HostFunction, ReductionSize);
    Rescue();
}

TTT std::ostream& operator<<(std::ostream& out, RA::CudaBridge<T>& obj)
{
    out << obj.GetStr();
    return out;
}

TTT template<typename F, typename ...A>
void RA::CudaBridge<T>::NONE::RunGPU(const dim3& FnGrid, const dim3& FnBlock, F&& Function, A&&... Args)
{
    Begin();
    //Function<<<FnGrid, FnBlock>>>(std::forward<A>(Args)...);
    //auto Error = cudaDeviceSynchronize();
    //if (Error)
    //    ThrowIt("CUDA Kernel: ", cudaGetErrorString(Error));

    cudaStream_t LvStream;
    auto Error = cudaStreamCreateWithFlags(&LvStream, cudaStreamNonBlocking);
    if (Error)
        ThrowIt("CUDA Stream Create Error: ", cudaGetErrorString(Error));

    Function<<<FnGrid, FnBlock, 0, LvStream>>>(std::forward<A>(Args)...);
    SvStreams.push_back(LvStream);

    //cudaStreamSynchronize(LoStream);
    Rescue();
}

TTT template<typename F, typename ...A>
void RA::CudaBridge<T>::NONE::RunCPU(F&& Function, A&&... Args)
{
    Begin();
    Function(std::forward<A>(Args)...);
    Rescue();
}

template<typename T>
template<typename F, typename ...A>
INL xvector<T> RA::CudaBridge<T>::NONE::RunMultiGPU(const uint FnSize, F&& Function, A && ...Args)
{
    Begin();
    xvector<xp<RA::CudaBridge<T>>> DeviceOutputs;

    uint LnSize = (FnSize / GetDeviceCount());
    const auto LnStandardDeviceSize = LnSize;

    auto [LvGrid, LvBlock] = RA::Host::GetDimensions3D(LnSize);

    auto Error = cudaDeviceSynchronize();
    if (Error)
        ThrowIt("CUDA Kernel: ", cudaGetErrorString(Error));

    //typedef int* Arr;
    //Arr vals = new int[5];

    auto LvStreams = std::make_shared<CUstream_st[]>(SnDeviceCount);
    auto LvEvents = std::make_shared<CUevent_st[]>(SnDeviceCount);
    for (int i = 0; i < SnDeviceCount; i++)
    {
        SetDeviceIdx(i);
        cudaStream_t LoStream = &LvStreams[i];
        Error = cudaStreamCreate(&LoStream);
        if (Error)
            ThrowIt("CUDA Stream Create Error: ", cudaGetErrorString(Error));
        cudaEvent_t LoEvent = &LvEvents[i];
        Error = cudaEventCreate(&LoEvent);
        if (Error)
            ThrowIt("CUDA Event Create Error: ", cudaGetErrorString(Error));
    }

    LnSize = LnStandardDeviceSize;
    for (auto i = 0; i < GetDeviceCount(); i++)
    {
        CudaBridge<>::SetDeviceIdx(i);
        if (i == GetDeviceCount() - 1)
        {
            LnSize += FnSize % SnDeviceCount;
            std::tie(LvGrid, LvBlock) = RA::Host::GetDimensions3D(LnSize);
        }

        cudaStream_t LoStream = &LvStreams[i];
        //auto& LoStream = (i == 0) ? LoStream1 : LoStream2;
        Function << <LvGrid, LvBlock, i, LoStream >> > (
            LnStandardDeviceSize * i,
            LnSize,
            std::forward<A>(Args)...);

        Error = cudaEventRecord((cudaEvent_t) &LvEvents[i], (cudaStream_t) &LvStreams[i]);
        if (Error)
            ThrowIt("CUDA Event Record Error: ", cudaGetErrorString(Error));
    }

    const auto LnDCount = SnDeviceCount.load();
    for (auto i = 0; i < LnDCount; i++)
    {
        Nexus<>::AddJob([](cudaEvent_t& FoEvent, const int i) {
            CudaBridge<>::SetDeviceIdx(i);
            auto Error = cudaEventSynchronize(FoEvent);
            if (Error)
                printf("CUDA Event Sync Error: %s\n", cudaGetErrorString(Error));
            }, LvEvents[i], i);
    }
    Nexus<>::WaitAll();

    for (auto i = 0; i < GetDeviceCount(); i++)
    {
        CudaBridge<>::SetDeviceIdx(i);
        Error = cudaDeviceSynchronize();
        if (Error)
            ThrowIt("CUDA Kernel: ", cudaGetErrorString(Error));
    }

    Rescue();
}

TTT template<typename F, typename ...A>
RA::CudaBridge<T> RA::CudaBridge<T>::ARRAY::RunGPU(const RA::Allocate& FoAllocate, const dim3& FnGrid, const dim3& FnBlock, F&& Function, A&&... Args)
{
    Begin();
    RA::CudaBridge<T> DeviceOutput(FoAllocate);
    DeviceOutput.AllocateHost();
    DeviceOutput.AllocateDevice();
    DeviceOutput.CopyHostToDeviceAsync();
    DeviceOutput.SyncStream();

    Function<<<FnGrid, FnBlock, 0, DeviceOutput.GetStream()>>>(DeviceOutput.GetDevice(), std::forward<A>(Args)...);

    return DeviceOutput;
    Rescue();
}

TTT template<typename F, typename ...A>
xvector<xp<RA::CudaBridge<T>>>
RA::CudaBridge<T>::ARRAY::RunUnjoinedMultiGPU(
    const RA::Allocate& FoAllocate, 
    const dim3& FnGrid, 
    const dim3& FnBlock, F&& Function, A&&... Args)
{
    Begin();
    xvector<xp<CudaBridge<T>>> DeviceOutputs;

    ThrowIt("Unbuilt");
    return DeviceOutputs;
    Rescue();
}

TTT template<typename F, typename ...A>
xvector<T> RA::CudaBridge<T>::ARRAY::RunMultiGPU(const Allocate& FoAllocate, F&& Function, A && ...Args)
{
    Begin();
    xvector<xp<RA::CudaBridge<T>>> DeviceOutputs;

    uint LnSize = (FoAllocate.GetLength() / GetDeviceCount());
    const auto LnStandardDeviceSize = LnSize;

    auto [LvGrid, LvBlock] = RA::Host::GetDimensions3D(LnSize);
    for (auto i = 0; i < GetDeviceCount(); i++)
    {
        CudaBridge<>::SetDeviceIdx(i);
        DeviceOutputs.Add(RA::MakeShared<RA::CudaBridge<T>>());
        if (i == GetDeviceCount() - 1)
            LnSize += LnStandardDeviceSize % SnDeviceCount;
        DeviceOutputs[i].AllocateHost(LnSize);
        DeviceOutputs[i].AllocateDevice();
        DeviceOutputs[i].CopyHostToDeviceAsync();
    }

    auto Error = cudaDeviceSynchronize();
    if (Error)
        ThrowIt("CUDA Kernel: ", cudaGetErrorString(Error));

    auto LvStreams = std::make_shared<cudaStream_t[]>(SnDeviceCount);
    auto LvEvents  = std::make_shared<cudaEvent_t[]>(SnDeviceCount);
    for (int i = 0; i < SnDeviceCount; i++)
    {
        SetDeviceIdx(i);
        Error = cudaStreamCreate((cudaStream_t*) &LvStreams[i]);
        if (Error)
            ThrowIt("CUDA Stream Create Error: ", cudaGetErrorString(Error));
        Error = cudaEventCreate((cudaEvent_t*) &LvEvents[i]);
        if (Error)
            ThrowIt("CUDA Event Create Error: ", cudaGetErrorString(Error));
    }

    LnSize = LnStandardDeviceSize;
    for (auto i = 0; i < GetDeviceCount(); i++)
    {
        CudaBridge<>::SetDeviceIdx(i);
        if (i == GetDeviceCount() - 1)
        {
            LnSize += FoAllocate.GetLength() % SnDeviceCount;
            std::tie(LvGrid, LvBlock) = RA::Host::GetDimensions3D(LnSize);
        }

        //auto& LoStream = (i == 0) ? LoStream1 : LoStream2;
        Function<<<LvGrid, LvBlock, i, LvStreams[i]>>>(
            LnStandardDeviceSize * i,
            LnSize,
            DeviceOutputs[i].GetDevice(),
            std::forward<A>(Args)...);

        Error = cudaEventRecord(LvEvents[i], LvStreams[i]);
        if (Error)
            ThrowIt("CUDA Event Record Error: ", cudaGetErrorString(Error));
    }

    const auto LnDCount = SnDeviceCount.load();
    for (auto i = 0; i < LnDCount; i++)
    {
        Nexus<>::AddJob([](cudaEvent_t& FoEvent, const int i) {
            CudaBridge<>::SetDeviceIdx(i);
            auto Error = cudaEventSynchronize(FoEvent);
            if (Error)
                printf("CUDA Event Sync Error: %s\n", cudaGetErrorString(Error));
            }, LvEvents[i], i);
    }
    Nexus<>::WaitAll();

    for (auto i = 0; i < GetDeviceCount(); i++)
        DeviceOutputs[i].SyncStream();
    //cudaStreamSynchronize(LoStream1);
    //cudaStreamSynchronize(LoStream2);

    for (auto i = 0; i < GetDeviceCount(); i++)
    {
        CudaBridge<>::SetDeviceIdx(i);
        Error = cudaDeviceSynchronize();
        if (Error)
            ThrowIt("CUDA Kernel: ", cudaGetErrorString(Error));
    }

    for (auto i = 0; i < GetDeviceCount(); i++)
        DeviceOutputs[i].CopyDeviceToHostAsync();

    for (auto i = 0; i < GetDeviceCount(); i++)
    {
        CudaBridge<>::SetDeviceIdx(i);
        Error = cudaDeviceSynchronize();
        if (Error)
            ThrowIt("CUDA Kernel: ", cudaGetErrorString(Error));
        cudaStreamDestroy((cudaStream_t) LvStreams[i]);
        cudaEventDestroy((cudaEvent_t) LvEvents[i]);
    }

    xvector<T> ReturnVec;
    for (auto i = 0; i < GetDeviceCount(); i++)
    {
        for (auto& Val : DeviceOutputs[i])
            ReturnVec << Val;
    }

    return ReturnVec;
    Rescue();
}


TTT template<typename F, typename ...A>
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
void RA::CudaBridge<T>::SyncAll()
{
    Begin();
    cudaDeviceSynchronize();
    for (uint i = 0; i < SvStreams.size(); i++)
    {
        auto Error = cudaStreamDestroy(SvStreams[i]);
        if(Error)
            ThrowIt("CUDA cudaStreamDestroy Error: ", cudaGetErrorString(Error));
    }
    SvStreams.clear();
    Rescue();
}

TTT bool RA::CudaBridge<T>::SameArrays(const RA::CudaBridge<T>& One, const RA::CudaBridge<T>& Two)
{
    if (One.Size() != Two.Size())
        return false;
    try
    {
        for (uint i = 0; i < One.Size(); i++)
        {
            if (One.MvHost[i] != Two.MvHost[i])
                return false;
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}

TTT bool RA::CudaBridge<T>::SameArrays(const xvector<T>& One, const xvector<T>& Two)
{
    if (One.Size() != Two.Size())
        return false;
    try
    {
        for (uint i = 0; i < One.Size(); i++)
        {
            if (One.At(i) != Two.At(i))
                return false;
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}

TTT RA::CudaBridge<T> RA::CudaBridge<T>::SumArraysIndicesCPU(
    const RA::CudaBridge<T>& One, const RA::CudaBridge<T>& Two)
{
    Begin();
    RA::CudaBridge<T> LoCombinedArray(One.Size());
    LoCombinedArray.AllocateHost();
    LoCombinedArray.ZeroHostData();

    for (uint i = 0; i < One.Size(); i++)
        LoCombinedArray.MvHost[i] += One.MvHost[i] + Two.MvHost[i];
    return LoCombinedArray;
    Rescue();
}

TTT void RA::CudaBridge<T>::CopyHostData(const T* FnArray)
{
    AllocateHost();
    for (uint i = 0; i < MnLeng; i++)
        MvHost[i] = FnArray[i];
}

TTT void RA::CudaBridge<T>::CopyHostData(const RA::SharedPtr<T[]>& FnArray)
{
    Begin();
    AllocateHost();
    for (uint i = 0; i < MnLeng; i++)
        MvHost[i] = FnArray[i];
    Rescue();
}

#endif