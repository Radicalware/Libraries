#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include<iostream>
#include<sstream>
#include<math.h>
#include<string>
#include<memory>


template<typename T = int>
class CudaBridge
{
public:
    ~CudaBridge();
    CudaBridge(size_t FnSize);
    //CudaBridge(std::shared_ptr<T[]> FnArray, size_t FnSize);
    CudaBridge(T* FnArray, size_t FnSize);

    CudaBridge(const CudaBridge<T>& Other);
    CudaBridge(CudaBridge<T>&& Other) noexcept;

    CudaBridge(std::initializer_list<T> FnInit);

    void operator=(const CudaBridge<T>& Other);
    void operator=(CudaBridge<T>&& Other) noexcept;

    void AllocateHost();
    void AllocateDevice();

    void ZeroHostData();

    size_t Size() const { return MnSize; }
    size_t GetByteSize() const { return MnByteSize; }

    bool HasHostArray()   const { return bool(MaxHost.get()); }
    bool HasDeviceArray() const { return MaxDevice != nullptr; }

    void CopyHostToDevice();
    void CopyDeviceToHost();

    T& GetHost(size_t FnIdx);
    const T& GetHost(size_t FnIdx) const;
    std::shared_ptr<T[]> GetHost();
    const std::shared_ptr<T[]> GetHost() const;
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
    void CopyHostData(const std::shared_ptr<T[]> FnArray);
    void CopyDeviceData(const T* FnArray);

    enum MemType
    {
        IsHost,
        IsDevice
    };

    std::shared_ptr<T[]> MaxHost;
    T* MaxDevice = nullptr;

    size_t MnSize = 0;
    size_t MnByteSize = 0;
};

template<typename T>
CudaBridge<T>::~CudaBridge()
{
    if (MaxHost.get())
        MaxHost.reset();

    if (MaxDevice)
        cudaFree(MaxDevice);// todo: add error check
}

template<typename T>
CudaBridge<T>::CudaBridge(size_t FnSize) :
    MnSize(FnSize), MnByteSize(sizeof(T)* FnSize)
{
}

//template<typename T>
//CudaBridge<T>::CudaBridge(std::shared_ptr<T[]> FnArray, size_t FnSize) :
//    MaxHost(FnArray), MnSize(FnSize), MnByteSize(sizeof(T)* FnSize)
//{
//}

template<typename T>
CudaBridge<T>::CudaBridge(T* FnArray, size_t FnSize) :
    MaxHost(std::shared_ptr<int[]>(FnArray)), MnSize(FnSize), MnByteSize(sizeof(T)* FnSize)
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
        MaxHost[i] = Val;
        i++;
    }
}

template<typename T>
void CudaBridge<T>::operator=(const CudaBridge<T>& Other)
{
    MnSize = Other.Size();
    MnByteSize = sizeof(T) * MnSize;

    if (Other.HasHostArray())
    {
        AllocateHost();
        CopyHostData(Other.MaxHost);
    }

    //if (Other.HasDeviceArray())
    //{
    //    AllocateDevice();
    //    CopyDeviceData(Other.MaxDevice);
    //}
}

template<typename T>
void CudaBridge<T>::operator=(CudaBridge<T>&& Other) noexcept
{
    MnSize = Other.MnSize;
    MnByteSize = sizeof(T) * MnSize;

    if (Other.HasHostArray())
    {
        AllocateHost();
        std::swap(MaxHost, Other.MaxHost);
    }

    //if (Other.HasDeviceArray())
    //{
    //    AllocateDevice();
    //    std::swap(MaxDevice, Other.MaxDevice);
    //}
}


template<typename T>
void CudaBridge<T>::AllocateHost()
{
    if (MaxHost.get())
        MaxHost.reset();

    MaxHost = std::shared_ptr<T[]>(new T[MnSize]);
}

template<typename T>
void CudaBridge<T>::AllocateDevice()
{
    if (MaxDevice)
        cudaFree(MaxDevice);

    cudaMalloc((T**)&MaxDevice, MnByteSize);
}

template<typename T>
void CudaBridge<T>::ZeroHostData()
{
    std::fill(
        &MaxHost[0], // fill data from here
        &MaxHost[0] + MnSize, // to here
        0); // with null data
}

template<typename T>
void CudaBridge<T>::CopyHostToDevice()
{
    AllocateDevice();
    cudaMemcpy(MaxDevice, MaxHost.get(), MnByteSize, cudaMemcpyHostToDevice);
}

template<typename T>
void CudaBridge<T>::CopyDeviceToHost()
{
    AllocateHost();
    cudaMemcpy(MaxHost.get(), MaxDevice, MnByteSize, cudaMemcpyDeviceToHost);
}

template<typename T>
inline T& CudaBridge<T>::GetHost(size_t FnIdx)
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MaxHost[FnIdx];
}

template<typename T>
const T& CudaBridge<T>::GetHost(size_t FnIdx) const
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MaxHost[FnIdx];
}

template<typename T>
inline std::shared_ptr<T[]> CudaBridge<T>::GetHost()
{
    return MaxHost;
}

template<typename T>
const std::shared_ptr<T[]> CudaBridge<T>::GetHost() const
{
    return MaxHost;
}

template<typename T>
T* CudaBridge<T>::GetHostPtr()
{
    return MaxHost.get();
}

template<typename T>
const T* CudaBridge<T>::GetHostPtr() const
{
    return MaxHost.get();
}

template<typename T>
inline T& CudaBridge<T>::GetDevice(size_t FnIdx)
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MaxDevice[FnIdx];
}

template<typename T>
const T& CudaBridge<T>::GetDevice(size_t FnIdx) const
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MaxDevice[FnIdx];
}

template<typename T>
inline T* CudaBridge<T>::GetDevice()
{
    return MaxDevice;
}

template<typename T>
const T* CudaBridge<T>::GetDevice() const
{
    return MaxDevice;
}

template<typename T>
T* CudaBridge<T>::GetDevicePtr()
{
    return MaxDevice;
}

template<typename T>
const T* CudaBridge<T>::GetDevicePtr() const
{
    return MaxDevice;
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
        return Function(MaxHost.get(), MnSize);
    return Function(MaxHost.get(), Size);
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
    CudaFunction << <FnGrid, FnBlock >> > (Data.GetDevice(), ReturnedArray.GetDevice(), static_cast<int>(Size()));
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
    Function << <FnGrid, FnBlock >> > ();
}

template<typename T>
template<typename F>
void CudaBridge<T>::NONE::RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, const int FnValue)
{
    Function << <FnGrid, FnBlock >> > (FnValue);
}

template<typename T>
template<typename F>
void CudaBridge<T>::NONE::RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, const int* FnArray, const size_t Size)
{
    Function << <FnGrid, FnBlock >> > (FnArray, Size);
}

template<typename T>
template<typename F>
void CudaBridge<T>::NONE::RunGPU(F&& Function, const dim3& FnGrid, const dim3& FnBlock, CudaBridge<T>& Bridge)
{
    Bridge.CopyHostToDevice();
    Function << <FnGrid, FnBlock >> > (Bridge.GetDevice(), static_cast<int>(Bridge.Size()));
}

template<typename T>
template<typename F>
CudaBridge<T> CudaBridge<T>::ARRAY::RunGPU(F&& Function, const dim3 FnGrid, const dim3 FnBlock, CudaBridge<T>& FnHost1)
{
    CudaBridge<T> DeviceOutput(FnHost1.Size());
    DeviceOutput.AllocateDevice();

    Function << <FnGrid, FnBlock >> > (FnHost1.GetDevice(), DeviceOutput.GetDevice(), static_cast<int>(FnHost1.Size()));

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

    Function << <FnGrid, FnBlock >> > (FnHost1.GetDevice(), FnHost2.GetDevice(), DeviceOutput.GetDevice(), static_cast<int>(FnHost1.Size()));

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

    Function << <FnGrid, FnBlock >> > (FnHost1.GetDevice(), FnHost2.GetDevice(), DeviceOutput.GetDevice(), NX, NY);

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
        if (One.MaxHost[i] != Two.MaxHost[i])
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
        LoCombinedArray.MaxHost[i] += One.MaxHost[i] + Two.MaxHost[i];
    return LoCombinedArray;
}

template<typename T>
void CudaBridge<T>::CopyHostData(const T* FnArray)
{
    AllocateHost();
    for (size_t i = 0; i < MnSize; i++)
        MaxHost[i] = FnArray[i];
}

template<typename T>
void CudaBridge<T>::CopyHostData(const std::shared_ptr<T[]> FnArray)
{
    AllocateHost();
    for (size_t i = 0; i < MnSize; i++)
        MaxHost[i] = FnArray[i];
}

template<typename T>
void CudaBridge<T>::CopyDeviceData(const T* FnArray)
{
    AllocateDevice();
    for (size_t i = 0; i < MnSize; i++)
        MaxDevice[i] = FnArray[i];
}
