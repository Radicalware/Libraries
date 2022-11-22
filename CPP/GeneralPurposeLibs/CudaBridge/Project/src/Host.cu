
#include "Host.cuh"

#include <iostream>
using std::cout;
using std::endl;

__host__ void RA::Host::PrintDeviceStats()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);


    if (deviceCount == 0)
        printf("No CUDA support device found\n\n");
    else if (deviceCount > 1)
        printf("You have SLi Running!!\n\n");
    else
        printf("You have one video card running\n\n");

    printf("Number of devices:                               %d\n", deviceCount);

    const char* Line = "\n---------------------------------------------------------------------------\n";
    for (int devNo = 0; devNo < deviceCount; devNo++) {

        cudaDeviceProp iProp;
        printf(Line);
        cudaGetDeviceProperties(&iProp, devNo);
        printf("Device %d Model:                                 %s\n", devNo, iProp.name);
        printf("  Number of multiprocessors:                     %d\n", iProp.multiProcessorCount);
        printf("  clock rate:                                    %d\n", iProp.clockRate);
        printf("  Compute capability:                            %d.%d\n", iProp.major, iProp.minor);
        printf("  Total amount of global memory:                 %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
        printf("  Total amount of constant memory:               %4.2f KB\n", iProp.totalConstMem / 1024.0);
        printf("  Total amount of shared memory per block:       %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
        printf("  Total amount of shared memory per MP:          %4.2f KB\n", iProp.sharedMemPerMultiprocessor / 1024.0);
        printf("  Total number of registers available per block: %d\n", iProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", iProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", iProp.maxThreadsPerBlock);
        printf("  Maximum number of threads per multiprocessor:  %d\n", iProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of warps per multiprocessor:    %d\n", iProp.maxThreadsPerMultiProcessor / 32);

        printf("  Maximum Grid size:                            (%d,%d,%d)\n",
            iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);

        printf("  Maximum block dimension:                      (%d,%d,%d)\n",
            iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
    }
    printf(Line);
}


__host__ std::tuple<dim3, dim3> GetDimensionsND(const uint FnLeng, const uint FnDims)
{
    const auto LnVerticyCount = FnDims * 2;
    auto LnTotalRoot = static_cast<uint>(std::pow(FnLeng, 1.0 / LnVerticyCount)) + 1;
    const auto LnTotalSqr = std::pow(LnTotalRoot, LnVerticyCount);
    if (LnTotalSqr < FnLeng)
        LnTotalRoot++;

    if (LnTotalSqr > RA::Host::SnMaxBlockSize)
    {
        const auto LnThreadsOnGrid = (int)(static_cast<float>(FnLeng) / RA::Host::SnMaxBlockSize) + 1;
        auto GridRoot = static_cast<uint>(std::pow(LnThreadsOnGrid, 1.0 / FnDims));
        const auto LnGridSqr = std::pow(GridRoot, FnDims);
        if (LnGridSqr < FnLeng)
            GridRoot++;

        const auto LvBlock = dim3(16, 16, 4); // 1024 threads / block
        const auto LvGrid = dim3(GridRoot, GridRoot, GridRoot);
        return std::make_tuple(LvGrid, LvBlock);
    }
    else
    {
        const auto LvGrid  = dim3(LnTotalRoot, LnTotalRoot, LnTotalRoot);
        const auto LvBlock = dim3(LnTotalRoot, LnTotalRoot, LnTotalRoot);
        return std::make_tuple(LvGrid, LvBlock);
    }
}

__host__ std::tuple<dim3, dim3> RA::Host::GetDimensions3D(const uint FnLeng)
{
    return GetDimensionsND(FnLeng, 3);
}

__host__ std::tuple<dim3, dim3> RA::Host::GetDimensions2D(const uint FnLeng)
{
    return GetDimensionsND(FnLeng, 2);
}

__host__ std::tuple<dim3, dim3> RA::Host::GetDimensions1D(const uint FnLeng, const uint FnBlockSize)
{
    auto LnBlockSize = (FnBlockSize != 0) ? FnBlockSize : FnLeng / 2;
    LnBlockSize = (LnBlockSize > SnMaxBlockSize) ? SnMaxBlockSize : LnBlockSize;
    dim3 LvBlock(LnBlockSize);
    dim3 LvGrid((FnLeng / LvBlock.x) + 1);
    return std::make_tuple(LvGrid, LvBlock);
}
