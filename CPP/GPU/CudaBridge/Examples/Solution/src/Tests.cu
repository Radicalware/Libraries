#include "Tests.cuh"

#include "ImportCUDA.cuh"

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

struct DeviceConfig
{
    DefaultConstruct(DeviceConfig);

    int MnID = 0;
    uint32_t* MvDeviceData = nullptr;
    int MbCanReadPeer = 0;
    cudaStream_t MoStream;
};

void Test::Features()
{
    Begin();

    const auto LbUseP2P = true;

    int LnDeviceCount = 0;
    cudaGetDeviceCount(&LnDeviceCount);

    if (LnDeviceCount < 2)
    {
        cout << "Multi-GPU rig not found\n";
        return;
    }

    // buffer size
    uint32_t size = 1 << 26;

    // create structs and allocate data
    xvector<DeviceConfig> MvDevice;
    for (int i = 0; i < LnDeviceCount; i++)
    {
        MvDevice << DeviceConfig();
        MvDevice[i].MnID = i;
        cudaSetDevice(MvDevice[i].MnID);
        cudaMalloc((void**)&MvDevice[i].MvDeviceData, size);
    }

    // Check for P2P access (run once, stored as static vars)
    bool LbCanAccessPeer = true;
    if (LbUseP2P)
    {
        for (int i = 0; i < LnDeviceCount; i++)
        {
            for (int j = 0; j < LnDeviceCount; j++)
            {
                if (i == j)
                    continue;
                cudaDeviceCanAccessPeer(&MvDevice[i].MbCanReadPeer, MvDevice[i].MnID, MvDevice[j].MnID);
                printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", MvDevice[i].MnID, MvDevice[j].MnID, MvDevice[i].MbCanReadPeer);
                if (MvDevice[i].MbCanReadPeer == false)
                    LbCanAccessPeer = false;
            }
        }
    }

    // (run once, stored as static vars)
    if (LbUseP2P && LbCanAccessPeer)
    {
        // Enable P2P Access
        for (int i = 0; i < LnDeviceCount; i++)
        {
            for (int j = 0; j < LnDeviceCount; j++)
            {
                if (i == j)
                    continue;
                cudaSetDevice(MvDevice[i].MnID);
                cudaDeviceEnablePeerAccess(MvDevice[j].MnID, 0);
            }
        }
    }

    // Init Timing Data
    uint32_t repeat = 10;

    RA::Timer Time;
    // Init Stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    //Do a P2P memcpy
    for (int i = 0; i < repeat; ++i) {
        cudaMemcpyAsync(MvDevice[0].MvDeviceData, MvDevice[1].MvDeviceData, size, cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);
    // ~~ End of Test ~~

    // Check Timing & Performance

    double gb = size * repeat / (double)1e9;
    double bandwidth = gb * (Time.GetElapsedTimeMilliseconds() / 10);

    printf("Milliseconds: %llu\n", Time.GetElapsedTimeMilliseconds());
    printf("Unidirectional Bandwidth: %lf (GB/s)\n", bandwidth);

    if (LbUseP2P && LbCanAccessPeer)
    {
        // Enable P2P Access
        for (int i = 0; i < LnDeviceCount; i++)
        {
            for (int j = 0; j < LnDeviceCount; j++)
            {
                if (i == j)
                    continue;
                cudaSetDevice(MvDevice[i].MnID);
                cudaDeviceDisablePeerAccess(MvDevice[j].MnID);
            }
        }
    }

    // Clean Up
    for (int i = 0; i < LnDeviceCount; i++)
        cudaFree(MvDevice[i].MvDeviceData);

    cudaStreamDestroy(stream);
    Rescue();
}


// =============================================================================================================================

__global__ void Print3D()
{
    auto GID = RA::Device::GetThreadID();
    constexpr char LsSpacer2[] = "  ";
    constexpr char LsSpacer1[] = " ";
    auto LsSpacer = (GID < 10) ? LsSpacer2 : LsSpacer1;
    constexpr char LsDivider[] = RED "|" WHITE;
    RA::Device::Print(
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
    RA::Device::Print(
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

__global__ void PrintNum(xint Val)
{
    printf("PrintNum\n");
    xint Result = 0;
    for (xint i = 0; i < Val; i++)
        Result += i;

    printf("Result: %llu\n", Result);
}

void Test::PrintGridBlockThread()
{
    Begin();
    auto LvBlock  = dim3(3, 3);
    auto LvThread = dim3(3, 3);
    RA::CudaBridge<>::NONE::RunGPU(LvBlock, LvThread, Print2D);
    RA::CudaBridge<>::SyncAll();
    Rescue();
}

// =============================================================================================================================

__global__ void DelayedSumArraysIndicesGPU(int* c, const int* a, const int* b, const int size)
{
    const auto GID = RA::Device::GetThreadID();

    if (GID >= size)
        return;

    auto Val = RA::Device::Timer::SleepTicks(1000 * 1000 * 100);
    //printf("%llu\n", RA::Device::Timer::SleepTicks(1000 /** 1000 * 100*/));
    c[GID] = a[GID] + b[GID] + Val;
}

__global__ void DelayedSumArraysIndicesMultiGPU(const xint FnStart, const xint Size, int* c, const int* a, const int* b)
{
    const auto RID = RA::Device::GetThreadID(); // Return ID
    //printf("FnStart: %llu %llu\n", FnStart, Size);
    if (RID >= Size)
        return;
    auto GID = RID + FnStart; // Global ID

    auto Val = RA::Device::Timer::SleepTicks(1000 * 1000 * 100);
    //printf("%llu -- %llu\n", FnStart, RA::Device::Timer::SleepTicks(1000 /** 1000 * 100*/));
    c[RID] = a[GID] + b[GID] + Val;
}

__global__ void SumArraysIndicesGPU(int* c, const int* a, const int* b, const int size)
{
    const auto GID = RA::Device::GetThreadID();

    if (GID >= size)
        return;

    c[GID] = a[GID] + b[GID];
}

__host__ void SumArraysIndicesCPU(int* c, const int* a, const int* b, const int size)
{
    printf("SumArraysIndicesCPU Start\n");
    for (int i = 0; i < size; i++)
        c[i] = a[i] + b[i];
    printf("SumArraysIndicesCPU Done\n");
}

#define GetRandomInt() \
    ((int)(rand() & 0xFF))


// =============================================================================================================================

void Test::SumArrayIndiciesMultiStream(const xint FnOperations)
{
    Begin();
    const int ArraySize = FnOperations;

    auto HostArray1 = RA::CudaBridge<int>(ArraySize);
    HostArray1.AllocateHost();
    for (int& Val : HostArray1) Val = GetRandomInt();
    HostArray1.CopyHostToDeviceAsync();


    auto HostArray2 = RA::CudaBridge<int>(ArraySize);
    HostArray2.AllocateHost();
    for (int& Val : HostArray2) Val = GetRandomInt();
    HostArray2.CopyHostToDeviceAsync();

    Nexus<RA::CudaBridge<int>> LoNexus;

    RA::Timer Time;
    LoNexus.AddTask([&]() { return RA::CudaBridge<int>::ARRAY::RunCPU(
        HostArray2.GetAllocation(), SumArraysIndicesCPU, HostArray1.GetHost(), HostArray2.GetHost(), ArraySize); });
    LoNexus.AddTask([&]() { return RA::CudaBridge<int>::ARRAY::RunCPU(
        HostArray2.GetAllocation(), SumArraysIndicesCPU, HostArray1.GetHost(), HostArray2.GetHost(), ArraySize); });

    LoNexus.WaitAll();
    auto HostResult1 = LoNexus.Get(1).GetValue();
    auto HostResult2 = LoNexus.Get(2).GetValue();

    cout << "CPU: " << Time.GetElapsedTimeMilliseconds() << endl;
    Time.Reset();

    cout << "Same Host Arrays: " << RA::CudaBridge<>::SameArrays(HostResult1, HostResult2) << endl;

    Time.Reset();
    auto [LvGrid, LvBlock] = RA::Host::GetDimensions3D(ArraySize);
    auto DeviceResult1 = RA::CudaBridge<int>::ARRAY::RunGPU(HostArray1.GetAllocation(), LvGrid, LvBlock,
        SumArraysIndicesGPU, HostArray1.GetDevice(), HostArray2.GetDevice(), ArraySize);

    auto DeviceResult2 = RA::CudaBridge<int>::ARRAY::RunGPU(HostArray1.GetAllocation(), LvGrid, LvBlock,
        SumArraysIndicesGPU, HostArray1.GetDevice(), HostArray2.GetDevice(), ArraySize);

    RA::CudaBridge<>::SyncAll();
    DeviceResult1.CopyDeviceToHost();
    DeviceResult2.CopyDeviceToHost();

    for (int i = 0; i < 10; i++)
        cout << DeviceResult1[i] << " -- " << HostResult1[i] << endl;

    cout << "GPU: " << Time.GetElapsedTimeMilliseconds() << endl;
    Time.Reset();

    cout << "Same Device Arrays: " << RA::CudaBridge<>::SameArrays(DeviceResult1, DeviceResult2) << endl;

    cout << "Same Cross Arrays:  " << RA::CudaBridge<>::SameArrays(DeviceResult1, HostResult1) << endl;

    const auto LnPrintSize = 1;
    const auto LnVal = 5555;
    std::tie(LvGrid, LvBlock) = RA::Host::GetDimensions3D(LnPrintSize);
    RA::CudaBridge<int>::NONE::RunGPU(LvGrid, LvBlock, PrintNum, LnVal);
    RA::CudaBridge<int>::NONE::RunGPU(LvGrid, LvBlock, PrintNum, LnVal);
    RA::CudaBridge<>::SyncAll();
    
    Rescue();
}
void Test::SumArrayIndiciesMultiGPU(const xint FnOperations)
{
    Begin();

    const int ArraySize = FnOperations;

    auto HostArray1 = RA::CudaBridge<int>(ArraySize);
    HostArray1.AllocateHost();
    for (int& Val : HostArray1) Val = GetRandomInt();
    HostArray1.CopyHostToDeviceAsync();


    auto HostArray2 = RA::CudaBridge<int>(ArraySize);
    HostArray2.AllocateHost();
    for (int& Val : HostArray2) Val = GetRandomInt();
    HostArray2.CopyHostToDeviceAsync();

    //const int LvBlockSize = 128;
    //dim3 LvBlock(LvBlockSize);
    //dim3 LvGrid((HostArray1.Size() / LvBlock.x) + 1);
    RA::Timer Time;
    auto HostResult = RA::CudaBridge<int>::ARRAY::RunCPU(
        HostArray2.GetAllocation(), SumArraysIndicesCPU, HostArray1.GetHost(), HostArray2.GetHost(), ArraySize);
    cout << "CPU: " << Time.GetElapsedTimeMilliseconds() << endl;

    Time.Reset();
    xvector<int> DeviceResultMultiGPU = RA::CudaBridge<int>::ARRAY::RunMultiGPU(
        HostArray1.GetAllocation(),
        DelayedSumArraysIndicesMultiGPU, HostArray1.GetDevice(), HostArray2.GetDevice());
    RA::CudaBridge<>::SyncAll();
    cout << "Multi  GPU: " << Time.GetElapsedTimeMilliseconds() << endl;

    Time.Reset();
    auto [LvGrid, LvBlock] = RA::Host::GetDimensions3D(ArraySize);
    RA::CudaBridge<int> DeviceResultSingleGPU = RA::CudaBridge<int>::ARRAY::RunGPU(
        HostArray1.GetAllocation(), LvGrid, LvBlock, 
        DelayedSumArraysIndicesGPU, HostArray1.GetDevice(), HostArray2.GetDevice(), ArraySize);
    cout << "Single GPU: " << Time.GetElapsedTimeMilliseconds() << endl;

    cout << "Array Size: " << HostArray1.Size() << " " << ArraySize << endl;
    cout << LvGrid.x  << " " << LvGrid.y  << " " << LvGrid.z << endl;
    cout << LvBlock.x << " " << LvBlock.y << " " << LvBlock.z << endl;

    if (RA::CudaBridge<>::SameArrays(HostResult, DeviceResultSingleGPU))
        cout << "Single GPU Arrays are the same\n";
    else
        cout << "Single GPU Arrays are different\n";

    const auto  LvCPU = HostResult.GetVec().ReverseSort();
    const auto& LvMultiGPU = DeviceResultMultiGPU.ReverseSort();
    if (RA::CudaBridge<>::SameArrays(LvCPU, LvMultiGPU))
        cout << "Multi  GPU Arrays are the same\n";
    else
    {
        cout << "-------------" << endl;
        cout << "Multi  GPU Arrays are different\n";
        for (auto i = 0; i < LvMultiGPU.Size(); i++)
        {
            printf("C/D: %d/%d\n", LvCPU[i], LvMultiGPU[i]);
            if (i > 10)
                break;
        }
        cout << "-------------" << endl;
        for (auto i = LvMultiGPU.Size()-1; i > 0; i--)
        {
            printf("C/D: %d/%d\n", LvCPU[i], LvMultiGPU[i]);
            if (LvMultiGPU.Size() - 10 > i)
                break;
        }
        cout << "-------------" << endl;
        xint LnZeroCount = 0;
        for (xint i = 0; i < LvMultiGPU.Size(); i++)
        {
            if (LvMultiGPU[i] == 0)
                LnZeroCount++;
        }
        cout << "ZeroRatio: " << (double)LnZeroCount / LvMultiGPU.Size() << endl;
        if (LvMultiGPU.Size() == 0)
            cout << "Multi  GPU size is Zero\n";
    }

    //for (int i = 0; i < 10; i++)
    //    cout << HostResult[i] << " -- " << DeviceResult[i] << endl;
    Rescue();
}

// =============================================================================================================================


__global__ void BlockLevelMutex(int* FvOutData, const int* FvInDataArray, RA::Device::Mutex* FoMutexPtr, const xint FnSize)
{
    auto GID = RA::Device::GetThreadID();
    if (GID > FnSize)
        return;
    auto& FoMutex = *FoMutexPtr;
    auto Val = FvInDataArray[GID];
    FoMutex.BlockLock();
    if (Val > FvOutData[0])
    {
        atomicMax(&FvOutData[0], Val);
        atomicMax(&FvOutData[1], Val);
        atomicMax(&FvOutData[2], Val);
    }
    FoMutex.UnlockBlocks();
}

__device__ bool GreaterThan(const int Left, const int Right) {
    return Left > Right;
}

// =============================================================================================================================

auto GetTestData(xint FnOperationCount)
{
    RA::CudaBridge<int> FvInDataArray(FnOperationCount);
    FvInDataArray.AllocateHost();

    for (int& Item : FvInDataArray)
    {
        int A, B, C;
        A = GetRandomInt();
        B = GetRandomInt();
        C = GetRandomInt();
        Item = A * B * C;
    }

    cout << "CPU Test (Find Biggest Num)\n";
    RA::Timer Time;
    int LnMaxVal = 0;
    for (auto Item : FvInDataArray)
    {
        if (Item > LnMaxVal)
            LnMaxVal = Item;
    }
    cout << "Clocked MS: " << Time.GetElapsedTimeMilliseconds() << endl;

    return std::make_tuple(FvInDataArray, LnMaxVal);
}


template<typename F>
void TestMutex(F&& FfFunction, const xstring& FsFunctionName, const xint FnOperations)
{
    Begin();
    auto [LvData, LnMaxVal] = GetTestData(FnOperations);
    auto [LvGrid, LvBlock] = RA::Host::GetDimensions3D(FnOperations);

    cout << endl;
    cout << "Running: " << FsFunctionName << endl;
    cout << "Vertex Layout: "
        << LvGrid.x << '*' << LvGrid.y << '*' << LvGrid.z << " * "
        << LvBlock.x << '*' << LvBlock.y << '*' << LvBlock.z << endl;


    auto LoMutex = RA::CudaBridge<RA::Device::Mutex>(1, sizeof(RA::Device::Mutex));
    LoMutex.AllocateHost();
    LoMutex.AllocateDevice();
    LoMutex.CopyHostToDevice();

    LvData.CopyHostToDevice();

    RA::Timer Time;
    auto LvOutData = RA::CudaBridge<int>::ARRAY::RunGPU(RA::Allocate(3, sizeof(int)), LvGrid, LvBlock,
        FfFunction, LvData.GetDevice(), LoMutex.GetDevice(), FnOperations);
    auto LnElapsed = Time.GetElapsedTimeMilliseconds();

    cout << "Data Size: " << LvData.GetMallocSize() << endl;
    cout << "Elapsed Time Ms: " << LnElapsed << endl;

    for (auto var : LvOutData)
        cout << "val: " << var << endl;
    cout << ((LnMaxVal == LvOutData[0]) ? "Output Matches" : "Output Does Not Match") << endl;
    cout << "Returned Max Val: " << RA::FormatNum(LvOutData[0]) << endl;
    cout << "Actual   Max Val: " << RA::FormatNum(LnMaxVal) << "\n\n\n";
    Rescue();
}


// ------------------------------------------------------------------------------------

void Test::TestBlockMutex(const xint FnOperations)
{
    Begin();
    TestMutex(BlockLevelMutex, __CLASS__, FnOperations);
    Rescue();
}

// =============================================================================================================================

// https://forums.developer.nvidia.com/t/try-to-use-lock-and-unlock-in-cuda/50761
// nVidia suggests avoiding a Mutex in favor of reduction algorithms
// If you need locking, use a block lock and then negotiate for access within a threadblock using ordinary sync
//void Test::TestThreadMutex(const xint FnOperations)
//{
//    Begin();
//    TestMutex(ThreadLevelMutex, __CLASS__, FnOperations);
//    Rescue();
//}

// =============================================================================================================================
