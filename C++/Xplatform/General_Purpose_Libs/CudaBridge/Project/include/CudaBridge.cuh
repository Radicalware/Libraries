#pragma once

#include<sstream>
#include<math.h>
#include<string>
#include<memory>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

using uint = size_t;

template<typename T = int>
class CudaBridge
{
public:
    ~CudaBridge();
    CudaBridge(uint FnSize);

    CudaBridge(const CudaBridge<T>& Other);
    CudaBridge(CudaBridge<T>&& Other) noexcept;

    void operator=(const CudaBridge<T>& Other);
    void operator=(CudaBridge<T>&& Other) noexcept;

    void AllocateHost();
    void AllocateDevice();

    void ZeroHostData();

    uint Size() const { return MnSize; }
    uint GetByteSize() const { return MnByteSize; }

    bool HasHostArray()   const { return MaxHost   != nullptr; }
    bool HasDeviceArray() const { return MaxDevice != nullptr; }

    void CopyHostToDevice();
    void CopyDeviceToHost();

    T& GetHost(uint FnIdx);
    const T& GetHost(uint FnIdx) const;
    T* GetHost();
    const T* GetHost() const;

    T& GetDevice(uint FnIdx);
    const T& GetDevice(uint FnIdx) const;
    T* GetDevice();
    const T* GetDevice() const;

    static bool SameHostArrays(const CudaBridge<T>& One, const CudaBridge<T>& Two);
    static CudaBridge<T> SumArraysIndicesCPU(const CudaBridge<T>& One, const CudaBridge<T>& Two);

private:
    void CopyHostData(const T* FnArray);
    void CopyDeviceData(const T* FnArray);

    enum MemType
    {
        IsHost,
        IsDevice
    };

    uint MnSize = 0;
    uint MnByteSize = 0;

    T* MaxHost   = nullptr;
    T* MaxDevice = nullptr;
};

template<typename T>
CudaBridge<T>::~CudaBridge()
{
    if (MaxHost) 
        delete[] MaxHost;

    if (MaxDevice) 
        cudaFree(MaxDevice);// todo: add error check
}

template<typename T>
CudaBridge<T>::CudaBridge(uint FnSize) :
    MnSize(FnSize), MnByteSize(sizeof(T) * FnSize)
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
void CudaBridge<T>::operator=(const CudaBridge<T>& Other)
{
    MnSize = Other.Size();
    MnByteSize = sizeof(T) * MnSize;

    if (Other.HasHostArray())
    {
        AllocateHost();
        CopyHostData(Other.MaxHost);
    }

    if (Other.HasDeviceArray())
    {
        AllocateDevice();
        CopyDeviceData(Other.MaxDevice);
    }
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

    if (Other.HasDeviceArray())
    {
        AllocateDevice();
        std::swap(MaxDevice, Other.MaxDevice);
    }
}


template<typename T>
void CudaBridge<T>::AllocateHost()
{
    if (MaxHost)
        delete[] MaxHost;

    MaxHost = new T[MnSize];
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
    cudaMemcpy(MaxDevice, MaxHost, MnByteSize, cudaMemcpyHostToDevice);
}

template<typename T>
void CudaBridge<T>::CopyDeviceToHost()
{
    AllocateHost();
    cudaMemcpy(MaxHost, MaxDevice, MnByteSize, cudaMemcpyDeviceToHost);
}

template<typename T>
inline T& CudaBridge<T>::GetHost(uint FnIdx)
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MaxHost[FnIdx];
}

template<typename T>
const T& CudaBridge<T>::GetHost(uint FnIdx) const
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MaxHost[FnIdx];
}

template<typename T>
inline T* CudaBridge<T>::GetHost()
{
    return MaxHost;
}

template<typename T>
const T* CudaBridge<T>::GetHost() const
{
    return MaxHost;
}

template<typename T>
inline T& CudaBridge<T>::GetDevice(uint FnIdx)
{
    if (FnIdx >= MnSize)
        throw "CudaBridge<T>::Get >> FnIdx Is too big";

    return MaxDevice[FnIdx];
}

template<typename T>
const T& CudaBridge<T>::GetDevice(uint FnIdx) const
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
std::ostream& operator<<(std::ostream& out, CudaBridge<T>& obj)
{
    out << obj.GetStr();
    return out;
}

template<typename T>
bool CudaBridge<T>::SameHostArrays(const CudaBridge<T>& One, const CudaBridge<T>& Two)
{
    for (uint i = 0; i < One.Size(); i++)
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

    for (uint i = 0; i < One.Size(); i++)
        LoCombinedArray.MaxHost[i] += One.MaxHost[i] + Two.MaxHost[i];
    return LoCombinedArray;
}

template<typename T>
void CudaBridge<T>::CopyHostData(const T* FnArray)
{
    AllocateHost();
    for (uint i = 0; i < MnSize; i++)
        MaxHost[i] = FnArray[i];
}

template<typename T>
void CudaBridge<T>::CopyDeviceData(const T* FnArray)
{
    AllocateDevice();
    for (uint i = 0; i < MnSize; i++)
        MaxDevice[i] = FnArray[i];
}