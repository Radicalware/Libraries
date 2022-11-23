
add_compile_definitions("UsingNVCC")

enable_language(CUDA)
find_package(CUDA REQUIRED)


set(CMAKE_CUDA_ARCHITECTURES ${CUDA_GPU}) # Pascal GPUs (aka 1000 Series GPUs)


set(CUDA_INCLUDE_DIRS   "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/include")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe")
set(CUDA_LIB_PATH       "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64")

set(CMAKE_CUDA_STANDARD 17)
set(CPP_ARGS "${CPP_ARGS} /std:c++17")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CUDA_VERBOSE_BUILD ON)


include_directories(
    "${CUDALIB_DIR}/include"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/include"
)

set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64")

# set(CUDA_LIBRARIES "")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nvrtc.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cublas.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cublasLt.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cuda.lib")
# #list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cudadevrt.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cudart.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cudart_static.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cufft.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cufftw.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/curand.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cusolver.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cusolverMg.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/cusparse.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppc.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppial.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppicc.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppidei.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppif.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppig.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppim.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppist.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppisu.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nppitc.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/npps.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nvblas.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nvjpeg.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nvml.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/nvptxcompiler_static.lib")
# list(APPEND CUDA_LIBRARIES "${CUDA_DIR}/OpenCL.lib")

# link_libraries(${CUDA_LIBRARIES})

function(FinishConfiguringCUDA)
    set_property(TARGET ${THIS} PROPERTY CUDA_ARCHITECTURES ${CUDA_GPU})
    # target_compile_options(${THIS} PUBLIC 
    #     "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -rdc=true,-dlink>"
    # )
    if(${debug})
    message("NVCC Debug Symbols Added! (Run with NSight)")
    target_compile_options(${THIS} PUBLIC 
        $<$<COMPILE_LANGUAGE:CUDA>:  -g -G --expt-relaxed-constexpr --std=c++17 
        --compiler-options --relocatable-device-code=true,-rdc=true,--device-link,-dlink,-g,-G,--expt-relaxed-constexpr>
)
    else()
    target_compile_options(${THIS} PUBLIC 
        $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --std=c++17 
        --compiler-options --relocatable-device-code=true,-rdc=true,--device-link,-dlink,--expt-relaxed-constexpr>
        #$<$<CXX_COMPILER_ID:MSVC>: >
    )
    endif()

    # target_link_options(${THIS} PUBLIC 
    #     "$<$<LINKER_LANGUAGE:CXX>: /NODEFAULTLIB:library>"
    # )
endfunction()



# list(APPEND CUDA_NVCC_FLAGS "--relocatable-device-code=true")
# list(APPEND CUDA_NVCC_FLAGS "--device-link")

# list(APPEND CMAKE_CUDA_FLAGS "--relocatable-device-code=true")
# list(APPEND CMAKE_CUDA_FLAGS "--device-link")

# set(NVCC_LINK_FLAGS "${NVCC_LINK_FLAGS};/NODEFAULTLIB:library")

# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)

