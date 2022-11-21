#include "Tests.cuh"

#include "CudaImport.cuh"

#include "Macros.h"
#include "Timer.h"
#include "xstring.h"

#include "Device.cuh"
#include "Host.cuh"
#include "Mutex.cuh"
#include "CudaBridge.cuh"
#include "Timer.cuh"

#include <tuple>

// =============================================================================================================================

void Test::PrintDeviceStats()
{
    RA::Host::PrintDeviceStats();
}


// =============================================================================================================================

__global__ void Print3D()
{
    auto GID = RA::Device::GetThreadID();
    constexpr char LsSpacer2[] = "  ";
    constexpr char LsSpacer1[] = " ";
    auto LsSpacer = (GID < 10) ? LsSpacer2 : LsSpacer1;
    constexpr char LsDivider[] = RED "|" WHITE;
    Print(
        "GID:%s%d "
        "%s blockIdx.x : %d,   blockIdx.y : %d,   blockIdx.z : %d"
        "%s threadIdx.x : %d,   threadIdx.y : %d,   blockIdx.z : %d"
        "%s gridDim.x : %d,   gridDim.y : %d,  gridDim.z : %d\n",
        LsSpacer, GID,
        LsDivider, blockIdx.x, blockIdx.y, blockIdx.z,
        LsDivider, threadIdx.x, threadIdx.y, threadIdx.z,
        LsDivider, gridDim.x, gridDim.y, gridDim.z
    );
}

__global__ void Print2D()
{
    auto GID = RA::Device::GetThreadID();
    constexpr char LsSpacer2[] = "  ";
    constexpr char LsSpacer1[] = " ";
    auto LsSpacer = (GID < 10) ? LsSpacer2 : LsSpacer1;
    constexpr char LsDivider[] = RED "|" WHITE;
    Print(
        "GID:%s%d "
        "%s blockIdx.x : %d,   blockIdx.y : %d"
        "%s threadIdx.x : %d,   threadIdx.y : %d"
        "%s gridDim.x : %d,   gridDim.y : %d,  gridDim.z : %d\n",
        LsSpacer, GID,
        LsDivider, blockIdx.x, blockIdx.y,
        LsDivider, threadIdx.x, threadIdx.y,
        LsDivider, gridDim.x, gridDim.y, gridDim.z
    );
}

void Test::PrintGridBlockThread()
{
    Begin();
    auto LvBlock  = dim3(3, 3);
    auto LvThread = dim3(3, 3);
    RA::CudaBridge<>::NONE::RunGPU(LvBlock, LvThread, Print2D);
    Rescue();
}

// =============================================================================================================================

__global__ void SumArraysIndicesGPU(int* c, const int* a, const int* b, const int size)
{
    RA::Device::Timer Time;
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (index < size)
        c[index] = a[index] + b[index];

    Time.SleepTicks(1000);
    Time.PrintElapsedTimeSeconds();
}

__host__ void SumArraysIndicesCPU(int* c, const int* a, const int* b, const int size)
{
    for (int i = 0; i < size; i++)
        c[i] = a[i] + b[i];
}

#define GetRandomInt() \
    ((int)(rand() & 0xFF))


// =============================================================================================================================

void Test::SumArrayIndicies()
{
    Begin();
    const int ArraySize = 1 << 10;
    const int LvBlockSize = 128;

    auto HostArray1 = RA::CudaBridge<int>(ArraySize);
    HostArray1.AllocateHost();
    for (int& Val : HostArray1) Val = GetRandomInt();
    HostArray1.CopyHostToDevice();


    auto HostArray2 = RA::CudaBridge<int>(ArraySize);
    HostArray2.AllocateHost();
    for (int& Val : HostArray2) Val = GetRandomInt();
    HostArray2.CopyHostToDevice();

    dim3 LvBlock(LvBlockSize);
    dim3 LvGrid((HostArray1.Size() / LvBlock.x) + 1);

    auto DeviceResult = RA::CudaBridge<int>::ARRAY::RunGPU(
        HostArray1.GetAllocation(), LvGrid, LvBlock, 
        SumArraysIndicesGPU, HostArray1.GetDevice(), HostArray2.GetDevice(), ArraySize);

    auto HostResult   = RA::CudaBridge<int>::ARRAY::RunCPU(
        HostArray2.GetAllocation(), SumArraysIndicesCPU, HostArray1.GetHost(), HostArray2.GetHost(), ArraySize);

    if (RA::CudaBridge<>::SameHostArrays(HostResult, DeviceResult))
        cout << "Arrays are the same\n";
    else
        cout << "Arrays are different\n";

    for (int i = 0; i < 10; i++)
    {
        cout << HostResult[i] << " -- " << DeviceResult[i] << endl;
    }
    Rescue();
}

// =============================================================================================================================

__global__ void ThreadLevelMutex(int* FvOutData, const int* FvInDataArray, RA::Device::Mutex* FoMutexPtr, const uint FnSize)
{
    auto GID = RA::Device::GetThreadID();
    if (GID > FnSize)
        return;
    auto& LoMutex = *FoMutexPtr;
    auto Val = FvInDataArray[GID];
    bool LbLock = false;
    do
    {
        if (LbLock = LoMutex.BxGetLock())
        {
            while (FvOutData[0] < Val)
            {
                FvOutData[0] = Val;
                FvOutData[1] = Val;
                FvOutData[2] = Val;
                //printf("%d < %d\n", FvOutData[0], Val);
            }
        }
        LoMutex.Unlock();
    } while (LbLock);
    LoMutex.Unlock();
}

__device__ bool GreaterThan(const int Left, const int Right) {
    return Left > Right;
}

__global__ void BlockLevelMutex(int* FvOutData, const int* FvInDataArray, RA::Device::Mutex* FoMutexPtr, const uint FnSize)
{
    auto GID = RA::Device::GetThreadID();
    if (GID > FnSize)
        return;
    auto& LoMutex = *FoMutexPtr;

    auto Val = FvInDataArray[GID];
    LoMutex.BlockLock(blockIdx);
    atomicMax(&FvOutData[0], Val);
    atomicMax(&FvOutData[1], Val);
    atomicMax(&FvOutData[2], Val);
    LoMutex.Unlock();
}

// =============================================================================================================================

auto GetTestData(uint FnOperationCount)
{
    RA::CudaBridge<int> FvInDataArray(FnOperationCount);
    FvInDataArray.AllocateHost();

    int LnMaxVal = 0;
    for (int& Item : FvInDataArray)
    {
        int A, B, C;
        A = GetRandomInt();
        B = GetRandomInt();
        C = GetRandomInt();

        int LnValue = A * B * C;
        if (LnValue > LnMaxVal)
            LnMaxVal = LnValue;
        Item = LnValue;
    }
    //cout << "Max: " << LnMaxVal << endl;

    //for (auto var : FvInDataArray)
    //    cout << "val: " << var << endl;


    return std::make_tuple(FvInDataArray, LnMaxVal);
}


template<typename F>
void TestMutex(F&& FfFunction, const xstring& FsFunctionName, const uint FnOperations)
{
    Begin();
    auto LoMutex = RA::CudaBridge<RA::Device::Mutex>(1);
    LoMutex.AllocateDevice();
    LoMutex.CopyHostToDevice();

    auto [LvData, LnMaxVal] = GetTestData(FnOperations);
    LvData.CopyHostToDevice();

    auto LnLine = (FnOperations / 4 / 4 / 4) + 1;
    auto LvGrid = dim3(LnLine, LnLine, LnLine);
    auto LvBlock = dim3(LnLine, LnLine, LnLine);

    RA::Timer Time;
    auto FvOutData = RA::CudaBridge<int>::ARRAY::RunGPU(RA::Allocate(3, sizeof(int)), LvGrid, LvBlock,
        FfFunction, LvData.GetDevice(), LoMutex.GetDevice(), FnOperations);
    auto LnElapsed = Time.GetElapsedTimeMilliseconds();

    cout << endl;
    cout << "Ran: " << FsFunctionName << endl;
    Print("Vertex Layout\n");
    Print(LvGrid);
    cout << "Data Size: " << LvData.GetAllocationSize() << endl;
    cout << "Largest Int: " << FvOutData[0] << endl;
    cout << "Elapsed Time Ms: " << LnElapsed << endl;

    for (auto var : FvOutData)
        cout << "val: " << var << endl;
    cout << ((LnMaxVal == FvOutData[0]) ? "Output Matches" : "Output Does Not Match") << endl;
    cout << "Max Val: " << LnMaxVal << "\n\n\n";
    Rescue();
}


// ------------------------------------------------------------------------------------

void Test::TestBlockMutex(const uint FnOperations)
{
    Begin();
    TestMutex(BlockLevelMutex, __CLASS__, FnOperations);
    Rescue();
}

// =============================================================================================================================

void Test::TestThreadMutex(const uint FnOperations)
{
    Begin();
    TestMutex(ThreadLevelMutex, __CLASS__, FnOperations);
    Rescue();
}

// =============================================================================================================================