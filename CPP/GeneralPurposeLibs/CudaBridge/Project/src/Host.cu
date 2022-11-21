
#include "../include/Host.cuh"

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
