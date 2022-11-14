#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include "Memory.h"
#include "Iterator.h"

#include<iostream>
#include<vector>
#include<sstream>
#include<math.h>
#include<string>
#include<algorithm>


template<typename T = int>
class CudaBridge : public RA::Iterator<T>
{
public:
    ~CudaBridge();
    inline CudaBridge() {}

    CudaBridge(size_t FnSize);
    CudaBridge(size_t FnSize, size_t FnMemSize);
    CudaBridge(RA::SharedPtr<T[]> FnArray, size_t FnSize);
    CudaBridge(RA::SharedPtr<T[]> FnArray, size_t FnSize, size_t FnMemSize);
    CudaBridge(T* FnArray, size_t FnSize);
    CudaBridge(T* FnArray, size_t FnSize, size_t FnMemSize);

    CudaBridge(const CudaBridge<T>& Other);
    CudaBridge(CudaBridge<T>&& Other) noexcept;

    CudaBridge(std::initializer_list<T> FnInit);

    void Initialize(size_t FnSize);
    void Initialize(size_t FnSize, size_t FnMemSize);

    void operator=(const CudaBridge<T>& Other);
    void operator=(CudaBridge<T>&& Other) noexcept;

    void operator=(const std::vector<T>& Other);
    void operator=(std::vector<T>&& Other);

          T& operator[](const size_t Idx)       { return MvHost[Idx]; }
    const T& operator[](const size_t Idx) const { return MvHost[Idx]; }

    void AllocateHost();
    void AllocateDevice();

    void ZeroHostData();

    size_t Size() const { return MnSize; }
    size_t GetByteSize() const { return MnByteSize; }

    bool HasHostArray()   const { return bool(MvHost.get()); }
    bool HasDeviceArray() const { return MvDevice != nullptr; }

    void CopyHostToDevice();
    void CopyDeviceToHost();

    T& GetHost(size_t FnIdx);
    const T& GetHost(size_t FnIdx) const;
    RA::SharedPtr<T[]> GetHost();
    const RA::SharedPtr<T[]> GetHost() const;
    T* GetHostPtr();
    const T* GetHostPtr() const;

    T& GetDevice(size_t FnIdx);
    const T& GetDevice(size_t FnIdx) const;
    T* GetDevice();
    const T* GetDevice() const;
    T* GetDevicePtr();
    const T* GetDevicePtr() const;


    // =================================================================================================================
    // Reduction
    template<typename F>
    T ReductionCPU(F&& Function, size_t Size = 0) const;
    template<typename F1, typename F2>
    T ReductionGPU(F1&& CudaFunction, F2&& HostFunction, const dim3& FnGrid, const dim3& FnBlock, const int ReductionSize = 0) const;
    // =================================================================================================================
    // Run GPU to Void
    struct NONE
    {
        template<typename F>
        static void RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock);

        template<typename F>
        static void RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, const int FnValue);

        template<typename F>
        static void RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, const int* FnArray, const size_t Size);
        template<typename F>
        static void RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, CudaBridge<T>& Bridge);
    };
    // =================================================================================================================
    // Run Gpu to Bridge
    struct ARRAY
    {
        template<typename F>
        static CudaBridge<T> RunGPU(F&& Function, const dim3 FnGrid, const dim3 FnBlock, CudaBridge<T>& FnHost1);

        template<typename F>
        static CudaBridge<T> RunGPU(F&& Function, const dim3 FnGrid, const dim3 FnBlock,
            const CudaBridge<T>& FnHost1, const CudaBridge<T>& FnHost2);

        template<typename F>
        static CudaBridge<T> RunGPU(F&& Function, const dim3 FnGrid, const dim3 FnBlock,
            const CudaBridge<T>& FnHost1, const CudaBridge<T>& FnHost2, const int NX, const int NY);

        // -------------------------------------------------------------------------------------------------------------

        template<typename F>
        static CudaBridge<T> RunCPU(F&& Function, const CudaBridge<T>& FnHost1, const CudaBridge<T>& FnHost2);
    };
    // =================================================================================================================
    // Misc
    static bool SameHostArrays(const CudaBridge<T>& One, const CudaBridge<T>& Two);
    static CudaBridge<T> SumArraysIndicesCPU(const CudaBridge<T>& One, const CudaBridge<T>& Two);
    // -----------------------------------------------------------------------------------------------------------------

private:
    void CopyHostData(const T* FnArray);
    void CopyHostData(const RA::SharedPtr<T[]> FnArray);
    void CopyDeviceData(const T* FnArray);

    enum class MemType
    {
        IsHost,
        IsDevice
    };

    RA::SharedPtr<T[]> MvHost;
    T* MvDevice = nullptr;

    size_t MnSize = 0;
    size_t MnByteSize = 0;
};

template<typename T>
CudaBridge<T>::~CudaBridge()
{
    if (MvHost.get())
        MvHost.reset();

    if (MvDevice)
        cudaFree(MvDevice);// todo: add error check
}

template<typename T>
CudaBridge<T>::CudaBridge(size_t FnSize) :
    MnSize(FnSize), MnByteSize(sizeof(T)* FnSize)
{
}

template<typename T>
CudaBridge<T>::CudaBridge(size_t FnSize, size_t FnMemSize) :
    MnSize(FnSize), MnByteSize(FnMemSize)
{
}

template<typename T>
CudaBridge<T>::CudaBridge(RA::SharedPtr<T[]> FnArray, size_t FnSize) :
    MvHost(FnArray), MnSize(FnSize), MnByteSize(sizeof(T)* FnSize)
{
}

template<typename T>
CudaBridge<T>::CudaBridge(RA::SharedPtr<T[]> FnArray, size_t FnSize, size_t FnMemSize) :
    MvHost(FnArray), MnSize(FnSize), MnByteSize(FnMemSize)
{
}

template<typename T>
CudaBridge<T>::CudaBridge(T* FnArray, size_t FnSize) :
    MvHost(RA::SharedPtr<int[]>(FnArray)), MnSize(FnSize), MnByteSize(sizeof(T)* FnSize)
{
}

template<typename T>
CudaBridge<T>::CudaBridge(T* FnArray, size_t FnSize, size_t FnMemSize) :
    MvHost(RA::SharedPtr<int[]>(FnArray)), MnSize(FnSize), MnByteSize(FnMemSize)
{
}

template<typename T>
CudaBridge<T>::CudaBridge(const CudaBridge<T>& Other)
{
    *this = Other;
}

template<typename T>
CudaBridge<T>::CudaBridge(CudaBridge<T>&& Other) noexcept
{
    *this = std::move(Other);
}

template<typename T>
CudaBridge<T>::CudaBridge(std::initializer_list<T> FnInit)
{
    MnSize = FnInit.size();
    MnByteSize = MnSize * sizeof(T);
    AllocateHost();
    size_t i = 0;
    for (const T& Val : FnInit)
    {
        MvHost[i] = Val;
        i++;
    }
}

template<typename T>
void CudaBridge<T>::Initialize(size_t FnSize)
{
    MvHost = nullptr;
    if (MvDevice)
        delete[] MvDevice;
    T* MvDevice = nullptr;

    MnSize = FnSize;
    MnByteSize = (sizeof(T) * FnSize);
}

template<typename T>
void CudaBridge<T>::Initialize(size_t FnSize, size_t FnMemSize)
{
    MvHost = nullptr;
    if (MvDevice)
        delete[] MvDevice;
    MvDevice = nullptr;

    MnSize = FnSize;
    MnByteSize = FnMemSize;
}

template<typename T>
void CudaBridge<T>::operator=(const CudaBridge<T>& Other)
{
    MnSize = Other.Size();
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
}

template<typename T>
void CudaBridge<T>::operator=(CudaBridge<T>&& Other) noexcept
{
    MnSize = Other.MnSize;
    MnByteSize = Other.MnByteSize;

    if (Other.HasHostArray())
    {
        AllocateHost();
        MvHost = std::move(Other.MvHost);
    }
}

template<typename T>
void CudaBridge<T>::operator=(const std::vector<T>& Other)
{
    if (MnByteSize)
        throw "Cuda Bridge has 0 MnByteSize";

    MnSize = Other.size();

    AllocateHost();
    CopyHostData(Other.data());
}

template<typename T>
void CudaBridge<T>::operator=(std::vector<T>&& Other)
{
    if (MnByteSize)
        throw "Cuda Bridge has 0 MnByteSize";

    MnSize = Other.MnSize;

    AllocateHost();
    for (uint i = 0; i < MnSize; i++)
        std::swap_ranges(Other[i].begin(), Other[i].end(), &MvHost[i]);
}


template<typename T>
void CudaBridge<T>::AllocateHost()
{
    if (MvHost.get())
        MvHost.reset();

    MvHost = RA::SharedPtr<T[]>(new T[MnSize]);
    SetIterator(MvHost.Ptr(), &MnSize);
}

template<typename T>
void CudaBridge<T>::AllocateDevice()
{
    if (MvDevice)
        cudaFree(MvDevice);

    cudaMalloc((T**)&MvDevice, MnByteSize);
}

template<typename T>
void CudaBridge<T>::ZeroHostData()
{
    std::fill(
        &MvHost[0], // fill data from here
        &MvHost[0] + MnSize, // to here
        0); // with null data
}

template<typename T>
void CudaBridge<T>::CopyHostToDevice()
{
    AllocateDevice();
    cudaMemcpy(MvDevice, MvHost.get(), MnByteSize, cudaMemcpyHostToDevice);
}

template<typename T>
void CudaBridge<T>::CopyDeviceToHost()
{
    AllocateHost();
    cudaMemcpy(MvHost.get(), MvDevice, MnByteSize, cudaMemcpyDeviceToHost);
}

template<typename T>
inline T& CudaBridge<T>::GetHost(size_t FnIdx)
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MvHost[FnIdx];
}

template<typename T>
const T& CudaBridge<T>::GetHost(size_t FnIdx) const
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MvHost[FnIdx];
}

template<typename T>
inline RA::SharedPtr<T[]> CudaBridge<T>::GetHost()
{
    return MvHost;
}

template<typename T>
const RA::SharedPtr<T[]> CudaBridge<T>::GetHost() const
{
    return MvHost;
}

template<typename T>
T* CudaBridge<T>::GetHostPtr()
{
    return MvHost.get();
}

template<typename T>
const T* CudaBridge<T>::GetHostPtr() const
{
    return MvHost.get();
}

template<typename T>
inline T& CudaBridge<T>::GetDevice(size_t FnIdx)
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MvDevice[FnIdx];
}

template<typename T>
const T& CudaBridge<T>::GetDevice(size_t FnIdx) const
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MvDevice[FnIdx];
}

template<typename T>
inline T* CudaBridge<T>::GetDevice()
{
    return MvDevice;
}

template<typename T>
const T* CudaBridge<T>::GetDevice() const
{
    return MvDevice;
}

template<typename T>
T* CudaBridge<T>::GetDevicePtr()
{
    return MvDevice;
}

template<typename T>
const T* CudaBridge<T>::GetDevicePtr() const
{
    return MvDevice;
}

template<typename T>
template<typename F>
T CudaBridge<T>::ReductionCPU(F&& Function, size_t Size) const
{
    //CudaBridge<T> HostOutput(MnSize);
    //HostOutput.AllocateHost();
    //HostOutput.ZeroHostData();
    //HostOutput.AllocateDevice();
    //HostOutput.CopyHostToDevice();
    if (Size == 0)
        return Function(MvHost.get(), MnSize);
    return Function(MvHost.get(), Size);
}

template<typename T>
template<typename F1, typename F2>
T CudaBridge<T>::ReductionGPU(F1&& CudaFunction, F2&& HostFunction, const dim3& FnGrid, const dim3& FnBlock, const int ReductionSize) const
{
    CudaBridge<T> Data = *this;
    Data.AllocateDevice();
    Data.CopyHostToDevice();
    CudaBridge<T> ReturnedArray(Size());
    ReturnedArray.AllocateDevice();
    CudaFunction<<<FnGrid, FnBlock>>>(Data.GetDevice(), ReturnedArray.GetDevice(), static_cast<int>(Size()));
    cudaDeviceSynchronize();
    ReturnedArray.CopyDeviceToHost();
    if (ReductionSize == 0)
        return ReturnedArray.ReductionCPU(HostFunction, Size());
    return ReturnedArray.ReductionCPU(HostFunction, ReductionSize);
}

template<typename T>
std::ostream& operator<<(std::ostream& out, CudaBridge<T>& obj)
{
    out << obj.GetStr();
    return out;
}

template<typename T>
template<typename F>
void CudaBridge<T>::NONE::RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock)
{
    Function<<<FnGrid, FnBlock>>>();
}

template<typename T>
template<typename F>
void CudaBridge<T>::NONE::RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, const int FnValue)
{
    Function<<<FnGrid, FnBlock>>>(FnValue);
}

template<typename T>
template<typename F>
void CudaBridge<T>::NONE::RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, const int* FnArray, const size_t Size)
{
    Function<<<FnGrid, FnBlock>>>(FnArray, Size);
}

template<typename T>
template<typename F>
void CudaBridge<T>::NONE::RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, CudaBridge<T>& Bridge)
{
    Bridge.CopyHostToDevice();
    Function<<<FnGrid, FnBlock>>>(Bridge.GetDevice(), static_cast<int>(Bridge.Size()));
}

template<typename T>
template<typename F>
CudaBridge<T> CudaBridge<T>::ARRAY::RunGPU(F&& Function, const dim3 FnGrid, const dim3 FnBlock, CudaBridge<T>& FnHost1)
{
    CudaBridge<T> DeviceOutput(FnHost1.Size());
    DeviceOutput.AllocateDevice();

    Function<<<FnGrid, FnBlock>>>(FnHost1.GetDevice(), DeviceOutput.GetDevice(), static_cast<int>(FnHost1.Size()));

    cudaDeviceSynchronize();
    DeviceOutput.CopyDeviceToHost();

    return DeviceOutput;
}

template<typename T>
template<typename F>
CudaBridge<T> CudaBridge<T>::ARRAY::RunGPU(F&& Function, const dim3 FnGrid, const dim3 FnBlock,
    const CudaBridge<T>& FnHost1, const CudaBridge<T>& FnHost2)
{
    CudaBridge<T> DeviceOutput(FnHost1.Size());
    DeviceOutput.AllocateDevice();

    Function<<<FnGrid, FnBlock>>>(FnHost1.GetDevice(), FnHost2.GetDevice(), DeviceOutput.GetDevice(), static_cast<int>(FnHost1.Size()));

    cudaDeviceSynchronize();
    DeviceOutput.CopyDeviceToHost();

    return DeviceOutput;
}

template<typename T>
template<typename F>
CudaBridge<T> CudaBridge<T>::ARRAY::RunGPU(F&& Function, const dim3 FnGrid, const dim3 FnBlock,
    const CudaBridge<T>& FnHost1, const CudaBridge<T>& FnHost2, const int NX, const int NY)
{
    CudaBridge<T> DeviceOutput(FnHost1.Size());
    DeviceOutput.AllocateDevice();

    Function<<<FnGrid, FnBlock>>>(FnHost1.GetDevice(), FnHost2.GetDevice(), DeviceOutput.GetDevice(), NX, NY);

    cudaDeviceSynchronize();
    DeviceOutput.CopyDeviceToHost();

    return DeviceOutput;
}

template<typename T>
template<typename F>
CudaBridge<T> CudaBridge<T>::ARRAY::RunCPU(F&& Function, const CudaBridge<T>& FnHost1, const CudaBridge<T>& FnHost2)
{
    CudaBridge<T> HostResult(FnHost1.Size());
    HostResult.AllocateHost();
    Function(FnHost1.GetHostPtr(), FnHost2.GetHostPtr(), HostResult.GetHostPtr(), static_cast<int>(HostResult.Size()));
    return HostResult;
}

template<typename T>
bool CudaBridge<T>::SameHostArrays(const CudaBridge<T>& One, const CudaBridge<T>& Two)
{
    for (size_t i = 0; i < One.Size(); i++)
    {
        if (One.MvHost[i] != Two.MvHost[i])
            return false;
    }
    return true;
}

template<typename T>
CudaBridge<T> CudaBridge<T>::SumArraysIndicesCPU(
    const CudaBridge<T>& One, const CudaBridge<T>& Two)
{
    CudaBridge<T> LoCombinedArray(One.Size());
    LoCombinedArray.AllocateHost();
    LoCombinedArray.ZeroHostData();

    for (size_t i = 0; i < One.Size(); i++)
        LoCombinedArray.MvHost[i] += One.MvHost[i] + Two.MvHost[i];
    return LoCombinedArray;
}

template<typename T>
void CudaBridge<T>::CopyHostData(const T* FnArray)
{
    AllocateHost();
    for (size_t i = 0; i < MnSize; i++)
        MvHost[i] = FnArray[i];
}

template<typename T>
void CudaBridge<T>::CopyHostData(const RA::SharedPtr<T[]> FnArray)
{
    AllocateHost();
    for (size_t i = 0; i < MnSize; i++)
        MvHost[i] = FnArray[i];
}

template<typename T>
void CudaBridge<T>::CopyDeviceData(const T* FnArray)
{
    AllocateDevice();
    for (size_t i = 0; i < MnSize; i++)
        MvDevice[i] = FnArray[i];
}
