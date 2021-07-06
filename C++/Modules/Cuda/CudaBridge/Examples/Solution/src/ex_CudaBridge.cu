
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "CudaBridge.cuh"

using std::cout;
using std::endl;


__global__
void SumArraysIndicesGPU(int* a, int* b, int* c, int size) 
{
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (index < size)
        c[index] = a[index] + b[index];
    // note: no mem-lock required for CUDA (like mutex from pthread)
}

__host__
void SumArraysIndicesCPU(int* a, int* b, int* c, int size) 
{
    for (int i = 0; i < size; i++)
        c[i] = a[i] + b[i];
}

#define GetRandomInt() \
    (int)(rand() & 0xFF);

int main() 
{
    const int LoValueCount = 1 << 10;
    // const uint ByteSize = LoValueCount * sizeof(int);
    const int LoBlockSize = 128;

    CudaBridge<int> HostArray1(LoValueCount);
    CudaBridge<int> HostArray2(LoValueCount);

    HostArray1.AllocateHost();
    HostArray2.AllocateHost();

    for (uint i = 0; i < LoValueCount; i++)
        HostArray1.GetHost(i) = GetRandomInt();
    for (uint i = 0; i < LoValueCount; i++)
        HostArray2.GetHost(i) = GetRandomInt();

    auto CpuResult = CudaBridge<int>::SumArraysIndicesCPU(HostArray1, HostArray2);

    HostArray1.CopyHostToDevice();
    HostArray2.CopyHostToDevice();

    CudaBridge<int> DeviceOutput(LoValueCount);
    DeviceOutput.AllocateDevice();

    dim3 LoBlock(LoBlockSize);
    dim3 LoGrid((HostArray1.Size() / LoBlock.x) + 1);

    SumArraysIndicesGPU << <LoGrid, LoBlock >> > (
        HostArray1.GetDevice(),
        HostArray2.GetDevice(),
        DeviceOutput.GetDevice(),
        HostArray1.Size()
        );

    cudaDeviceSynchronize();
    DeviceOutput.CopyDeviceToHost();

    if (CudaBridge<int>::SameHostArrays(CpuResult, DeviceOutput))
        cout << "Arrays are the same\n";
    else
        cout << "Arrays are different\n";


    return 0;
}
