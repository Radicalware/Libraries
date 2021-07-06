

set(VCPKG_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/x64-windows/lib")
include("${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

set(CMAKE_CUDA_ARCHITECTURES 61) # Pascal GPUs (aka 1000 Series GPUs)
set(CUDA_INCLUDE_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include")

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

enable_language(CUDA)

include_directories("${VCPKG_ROOT}/installed/x64-windows/include") # use computer evn variable

include_directories(
        "${VCPKG_ROOT}/installed/x64-windows/include"

        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include"
        "${CUDALIB_DIR}/include"

        "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30037/include"
        "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30037/atlmfc/include"
        "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/VS/include"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/ucrt"
        "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/VS/UnitTest/include"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/um"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/shared"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/winrt"
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/cppwinrt"
        "C:/Program Files (x86)/Windows Kits/NETFXSDK/4.8/Include/um"
)

find_package(CUDA)

link_libraries(

    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nvrtc-prev/nvrtc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cublas.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cublasLt.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cuda.lib"
    # "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cudadevrt.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cudart.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cudart_static.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cufft.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cufftw.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/curand.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cusolver.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cusolverMg.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cusparse.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppial.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppicc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppidei.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppif.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppig.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppim.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppist.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppisu.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nppitc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/npps.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nvblas.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nvjpeg.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nvml.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nvptxcompiler_static.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/nvrtc.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/OpenCL.lib"
)
