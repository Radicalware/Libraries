
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "CudaBridge.cuh"

using std::cout;
using std::endl;
using uint = size_t;

/// Be sure to COPY the following
/// FROM:   "CUDA C/C++"        >> "Additional Include Directories" 
/// TO:     "VC++ Directories"  >> Include Directories

__global__
void SumArraysIndicesGPU(const int* a, const int* b, int* c, const int size)
{
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (index < size)
        c[index] = a[index] + b[index];
    // note: no mem-lock required for CUDA (like mutex from pthread)
}

__host__
void SumArraysIndicesCPU(const int* a, const int* b, int* c, const int size)
{
    for (int i = 0; i < size; i++)
        c[i] = a[i] + b[i];
}

#define GetRandomInt() \
    (int)(rand() & 0xFF);


int main()
{
    const int ArraySize = 1 << 10;

    // const uint ByteSize = ArraySize * sizeof(int);
    const int LoBlockSize = 128;

    CudaBridge<int> HostArray1(ArraySize);
    CudaBridge<int> HostArray2(ArraySize);

    HostArray1.AllocateHost();
    HostArray2.AllocateHost();

    for (uint i = 0; i < ArraySize; i++)
        HostArray1.GetHost(i) = GetRandomInt();
    for (uint i = 0; i < ArraySize; i++)
        HostArray2.GetHost(i) = GetRandomInt();

    HostArray1.CopyHostToDevice();
    HostArray2.CopyHostToDevice();

    dim3 LoBlock(LoBlockSize);
    dim3 LoGrid((HostArray1.Size() / LoBlock.x) + 1);

    auto DeviceResult = CudaBridge<int>::ARRAY::RunGPU(SumArraysIndicesGPU, LoGrid, LoBlock, HostArray1, HostArray2);
    auto HostResult   = CudaBridge<int>::ARRAY::RunCPU(SumArraysIndicesCPU, HostArray1, HostArray2);

    if (CudaBridge<>::SameHostArrays(HostResult, DeviceResult))
        cout << "Arrays are the same\n";
    else
        cout << "Arrays are different\n";

    return 0;
}
