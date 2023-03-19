

# add_compile_definitions("_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
# add_compile_definitions("JSON_HAS_RANGES=0")


set(UsingNVCC ON)

remove_definitions(-DUsingMSVC)
add_definitions(-DUsingNVCC)

#list(APPEND CMAKE_CUDA_FLAGS -DUsingNVCC)

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_GPU}) # Pascal GPUs (aka 1000 Series GPUs)

set(CUDA_INCLUDE_DIRS   "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/include")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe")
set(CUDA_LIB_PATH       "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64")
set(CUDA_DIR            "${CUDA_LIB_PATH}")


# -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -DJSON_HAS_RANGES=0

# set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(CUDA_VERBOSE_BUILD ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_DEFAULT 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_DEFAULT 20)

enable_language(CUDA)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_DEFAULT 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_DEFAULT 20)

# add_compile_definitions("UsingNVCC")
# set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS}  -DUsingNVCC")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUsingNVCC")

include_directories("${CUDA_INCLUDE_DIRS}")
link_directories("D:/AIE/vcpkg/installed/x64-windows/lib")
link_directories("${CUDA_LIB_PATH}")

function(ConfigCUDA BINARY)
    set_target_properties(${BINARY} PROPERTIES CUDA_ARCHITECTURES ${CUDA_GPU})
    set_target_properties(${BINARY} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(${debug})
        target_compile_options(${BINARY} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                               -std=c++20
                               -D"UsingNVCC=\"1\""
                               -rdc=true
                               -dlink
                               -g
                               -G
                               -diag-suppress=1394
                               -diag-suppress=3133
                               -diag-suppress=611
                               -diag-suppress=554
                               -diag-suppress=174
                               -diag-suppress=68
                               -arch=sm_61
                               >)

        # -Xcompiler="/DUsingNVCC"
        # 174 = allows expressions that have no affect (prevents annoying pop-ups on 3rd party code)
        # target_link_options(${BINARY} PRIVATE $<$<LINK_LANGUAGE:CUDA>:
        #                        >)

    else()
        target_compile_options(${BINARY} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                               -std=c++20
                               -D"UsingNVCC=\"1\""
                               -rdc=true
                               -dlink
                               -diag-suppress=1394
                               -diag-suppress=3133
                               -diag-suppress=611
                               -diag-suppress=554
                               -diag-suppress=174
                               -diag-suppress=68
                               -arch=sm_61
                               >)

        # -Xcompiler="/DUsingNVCC"
        # target_link_options(${BINARY} PRIVATE $<$<LINK_LANGUAGE:CUDA>:
        #                        >)
    endif()

endfunction()


# -----------------------------------------------------------------------------
# Notes Below
# -----------------------------------------------------------------------------

# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)


# -dlink
# --device-link
# -rdc=true
# --relocatable-device-code=true
# -diag-suppress=1394,3133 
# -DUsingNVCC

# target_compile_options(${BINARY} PUBLIC 
#     "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -rdc=true,-dlink>"
# )
# if(${debug})
# message("NVCC Debug Symbols Added! (Run with NSight)")
# target_compile_options(${BINARY} PUBLIC 
#     $<$<COMPILE_LANGUAGE:CUDA>:  -g -G --expt-relaxed-constexpr -std=c++20 -DUsingNVCC
#     --compiler-options --relocatable-device-code=true,-rdc=true,--device-link,-dlink,-g,-G,--expt-relaxed-constexpr,-DUsingNVCC,-std=c++20>
#     )
# else()
# target_compile_options(${BINARY} PUBLIC 
#     $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -std=c++20 -DUsingNVCC
#     --compiler-options --relocatable-device-code=true,-rdc=true,--device-link,-dlink,--expt-relaxed-constexpr,-DUsingNVCC,-std=c++20>
#     #$<$<CXX_COMPILER_ID:MSVC>: >
#     )
# endif()

# target_link_options(${BINARY} PUBLIC 
#     "$<$<LINKER_LANGUAGE:CXX>: /NODEFAULTLIB:library>"
# )

# target_compile_options(${BINARY} PUBLIC 
#     "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -rdc=true,-dlink>"
# ) 

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++20 -dlink -rdc=true -DUsingNVCC")
# list(APPEND CUDA_NVCC_FLAGS "--relocatable-device-code=true")
# list(APPEND CMAKE_CUDA_FLAGS "--relocatable-device-code=true")
# list(APPEND CMAKE_CUDA_FLAGS "--device-link")

# set(NVCC_LINK_FLAGS "${NVCC_LINK_FLAGS};/NODEFAULTLIB:library")


# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------