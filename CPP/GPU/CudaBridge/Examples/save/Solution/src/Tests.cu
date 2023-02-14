#include "Tests.cuh"

#include "Macros.h"

#include "CudaBridge.cuh"


// =============================================================================================================================


__global__ void PrintLargestIDX(const int* DataArray, const int* OutArray)
{
    uint Val = ThisGPU::GetThreadID();
}



void Test::FindMaxIdx()
{
    Begin();

    int LnOperations = 285;

    RA::CudaBridge<int> DataArray(LnOperations);
    DataArray.AllocateHost();
    for (int i = 0; i < LnOperations; i++)
        DataArray[i] = i * 2;
    DataArray.CopyHostToDevice();
    

    RA::CudaBridge<int> OutArray(2);
    OutArray.AllocateDevice();

    auto LoGrid  = dim3(2, 3);
    auto LoBlock = dim3(3, 2);

    RA::CudaBridge<int>::NONE::RunGPU(LoGrid, LoBlock, PrintLargestIDX, DataArray.GetDevice(), OutArray.GetDevice());
    cout << "Largest Int" << OutArray[0] << endl;
    
    Rescue();
}

// =============================================================================================================================

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


void Test::SumArrayIndicies()
{
    Begin();
    const int ArraySize = 1 << 10;

    // const uint ByteSize = ArraySize * sizeof(int);
    const int LoBlockSize = 128;

    RA::CudaBridge<int> HostArray1(ArraySize);
    RA::CudaBridge<int> HostArray2(ArraySize);

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

    auto DeviceResult = RA::CudaBridge<int>::ARRAY::RunGPU(
        HostArray1.GetAllocation(), LoGrid, LoBlock, SumArraysIndicesGPU, HostArray1.GetDevice(), HostArray2.GetDevice(), ArraySize);

    auto HostResult   = RA::CudaBridge<int>::ARRAY::RunCPU(
        HostArray2.GetAllocation(), SumArraysIndicesCPU, HostArray1.GetHost(), HostArray2.GetHost(), ArraySize);

    if (RA::CudaBridge<>::SameHostArrays(HostResult, DeviceResult))
        cout << "Arrays are the same\n";
    else
        cout << "Arrays are different\n";
    Rescue();
}

// =============================================================================================================================