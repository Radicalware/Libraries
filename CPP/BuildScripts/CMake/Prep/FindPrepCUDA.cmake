
add_compile_definitions("UsingNVCC")

set(CMAKE_CUDA_ARCHITECTURES 61) # Pascal GPUs (aka 1000 Series GPUs)
set(CUDA_INCLUDE_DIRS   "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/include")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe")
set(CUDA_LIB_PATH       "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

enable_language(CUDA)
find_package(CUDA REQUIRED)

include_directories(
    "${VCPKG_ROOT}/installed/x64-windows/include"
    "${CUDALIB_DIR}/include"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/include"
)

link_libraries(

    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nvrtc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cublas.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cublasLt.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cuda.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cudadevrt.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cudart.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cudart_static.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cufft.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cufftw.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/curand.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cusolver.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cusolverMg.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/cusparse.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppial.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppicc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppidei.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppif.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppig.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppim.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppist.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppisu.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nppitc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/npps.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nvblas.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nvjpeg.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nvml.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/nvptxcompiler_static.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64/OpenCL.lib"
)


#set_target_properties(${THIS} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)