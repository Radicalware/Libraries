
#include "Host.cuh"

#include <iostream>
using std::cout;
using std::endl;

uint RA::Host::SnThreadsPerBlock = 0;
dim3 RA::Host::SvBlock3D = 0;
dim3 RA::Host::SvBlock2D = 0;
uint RA::Host::SnThreadsPerWarp = 0;

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
        cudaGetDeviceProperties(&iProp, devNo);
        printf(Line);
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


__host__ std::tuple<dim3, dim3> RA::Host::GetDimensions3D(const uint FnLeng)
{
    auto GetThreadCube = [](const uint FnCubeSize)->dim3
    {
        uint LnXY = 1;
        uint LnZ = FnCubeSize;
        uint LnNewZ = LnZ;
        while (LnXY < LnZ)
        {
            LnNewZ /= 2;
            const auto LnXYThreadCount = FnCubeSize / LnNewZ;
            const auto LnNewXY = static_cast<uint>(std::sqrt(LnXYThreadCount));
            if (LnNewXY * LnNewXY * LnNewZ != FnCubeSize)
                continue;
            LnXY = LnNewXY;
            LnZ = LnNewZ;
        }
        return dim3(LnXY, LnXY, LnZ);
    };

    if (!SnThreadsPerBlock)
        RA::Host::PopulateStaticNums();
    if (SnThreadsPerBlock > (FnLeng / 2))
        return RA::Host::GetDimensions1D(FnLeng);
    if (SnThreadsPerBlock > (FnLeng / 4))
        return RA::Host::GetDimensions2D(FnLeng);
    if (SvBlock3D.x == 0)
        SvBlock3D = GetThreadCube(SnThreadsPerBlock);

    const auto LvBlock = SvBlock3D;
    const auto LnRemainder = (FnLeng % SnThreadsPerBlock) ? 1 : 0;
    const auto LvGrid = GetThreadCube(FnLeng / SnThreadsPerBlock + LnRemainder);
    return std::make_tuple(LvGrid, LvBlock);
}

__host__ std::tuple<dim3, dim3> RA::Host::GetDimensions2D(const uint FnLeng)
{
    auto GetThreadSquare = [](const uint FnSquareSize)->dim3
    {
        uint LnX = 1;
        uint LnY = FnSquareSize;
        uint LnNewY = LnY;
        while (LnX < LnY)
        {
            LnNewY /= 2;
            const auto LnNewX = FnSquareSize / LnNewY;
            if (LnNewX * LnNewY != FnSquareSize)
                continue;
            LnX = LnNewX;
            LnY = LnNewY;
        }
        return dim3(LnX, LnY);
    };

    if (!SnThreadsPerBlock)
        RA::Host::PopulateStaticNums();
    if (SnThreadsPerBlock > (FnLeng / 2))
        return RA::Host::GetDimensions1D(FnLeng);
    if(SvBlock2D.x == 0)
        SvBlock2D = GetThreadSquare(SnThreadsPerBlock);

    const auto LvBlock = SvBlock2D;
    const auto LnRemainder = (FnLeng % SnThreadsPerBlock) ? 1 : 0;
    const auto LvGrid = GetThreadSquare(FnLeng / SnThreadsPerBlock + LnRemainder);
    return std::make_tuple(LvGrid, LvBlock);
}

__host__ std::tuple<dim3, dim3> RA::Host::GetDimensions1D(const uint FnLeng)
{
    if (!SnThreadsPerBlock)
        RA::Host::PopulateStaticNums();
    if (FnLeng < SnThreadsPerWarp)
    {
        const dim3 LvBlock = FnLeng;
        const dim3 LvGrid = 1;
        return std::make_tuple(LvGrid, LvBlock);
    }
    const dim3 LvBlock(SnThreadsPerWarp);
    dim3 LvGrid = (FnLeng / LvBlock.x);
    if (LvGrid.x * LvBlock.x < FnLeng)
        LvGrid.x++;
    return std::make_tuple(LvGrid, LvBlock);
}

__host__ void RA::Host::PopulateStaticNums()
{
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, 0);
    SnThreadsPerBlock = iProp.maxThreadsPerBlock;
    SnThreadsPerWarp = iProp.warpSize;
}