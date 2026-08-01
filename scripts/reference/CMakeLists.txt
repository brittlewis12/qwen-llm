cmake_minimum_required(VERSION 3.14...3.28)

project(qwen_llm_reference_oracles LANGUAGES C CXX ASM)

if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "configure with -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_AVX OFF CACHE BOOL "" FORCE)
set(GGML_AVX2 OFF CACHE BOOL "" FORCE)
set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_BF16 OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_VBMI OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_VNNI OFF CACHE BOOL "" FORCE)
set(GGML_AVX_VNNI OFF CACHE BOOL "" FORCE)
set(GGML_BMI2 OFF CACHE BOOL "" FORCE)
set(GGML_F16C OFF CACHE BOOL "" FORCE)
set(GGML_FMA OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_SSE42 OFF CACHE BOOL "" FORCE)
set(GGML_AMX_BF16 OFF CACHE BOOL "" FORCE)
set(GGML_AMX_INT8 OFF CACHE BOOL "" FORCE)
set(GGML_AMX_TILE OFF CACHE BOOL "" FORCE)

function(dsv4_metadata_value output value)
    if("${value}" STREQUAL "")
        set(${output} "<default>" PARENT_SCOPE)
    else()
        string(REPLACE "\n" " " value "${value}")
        set(${output} "${value}" PARENT_SCOPE)
    endif()
endfunction()

dsv4_metadata_value(DSV4_C_FLAGS "${CMAKE_C_FLAGS}")
dsv4_metadata_value(DSV4_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
dsv4_metadata_value(DSV4_C_COMPILER_TARGET "${CMAKE_C_COMPILER_TARGET}")
dsv4_metadata_value(DSV4_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
dsv4_metadata_value(DSV4_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
dsv4_metadata_value(DSV4_CXX_COMPILER_TARGET "${CMAKE_CXX_COMPILER_TARGET}")
dsv4_metadata_value(DSV4_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
dsv4_metadata_value(DSV4_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
dsv4_metadata_value(DSV4_OSX_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}")
dsv4_metadata_value(DSV4_OSX_DEPLOYMENT_TARGET "${CMAKE_OSX_DEPLOYMENT_TARGET}")
dsv4_metadata_value(DSV4_OSX_SYSROOT "${CMAKE_OSX_SYSROOT}")

file(WRITE "${CMAKE_BINARY_DIR}/dsv4-reference-toolchain.txt"
    "cmake_version=${CMAKE_VERSION}\n"
    "generator=${CMAKE_GENERATOR}\n"
    "c_compiler=${CMAKE_C_COMPILER}\n"
    "c_compiler_id=${CMAKE_C_COMPILER_ID}\n"
    "c_compiler_target=${DSV4_C_COMPILER_TARGET}\n"
    "c_compiler_version=${CMAKE_C_COMPILER_VERSION}\n"
    "c_flags=${DSV4_C_FLAGS}\n"
    "c_flags_release=${DSV4_C_FLAGS_RELEASE}\n"
    "cxx_compiler=${CMAKE_CXX_COMPILER}\n"
    "cxx_compiler_id=${CMAKE_CXX_COMPILER_ID}\n"
    "cxx_compiler_target=${DSV4_CXX_COMPILER_TARGET}\n"
    "cxx_compiler_version=${CMAKE_CXX_COMPILER_VERSION}\n"
    "cxx_flags=${DSV4_CXX_FLAGS}\n"
    "cxx_flags_release=${DSV4_CXX_FLAGS_RELEASE}\n"
    "exe_linker_flags=${DSV4_EXE_LINKER_FLAGS}\n"
    "exe_linker_flags_release=${DSV4_EXE_LINKER_FLAGS_RELEASE}\n"
    "osx_architectures=${DSV4_OSX_ARCHITECTURES}\n"
    "osx_deployment_target=${DSV4_OSX_DEPLOYMENT_TARGET}\n"
    "osx_sysroot=${DSV4_OSX_SYSROOT}\n"
    "system_name=${CMAKE_SYSTEM_NAME}\n"
    "system_processor=${CMAKE_SYSTEM_PROCESSOR}\n"
    "system_version=${CMAKE_SYSTEM_VERSION}\n"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_subdirectory("${LLAMA_CPP_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml")

add_executable(dsv4_llama_cpp_cpu_oracle dsv4_llama_cpp_cpu_oracle.cpp)
target_compile_features(dsv4_llama_cpp_cpu_oracle PRIVATE cxx_std_17)
target_link_libraries(dsv4_llama_cpp_cpu_oracle PRIVATE ggml)
set_target_properties(dsv4_llama_cpp_cpu_oracle PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin"
)
