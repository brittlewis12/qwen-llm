cmake_minimum_required(VERSION 3.20)
project(glm53_nextn_probe LANGUAGES CXX)

if(NOT LLAMA_CPP_DIR OR NOT LLAMA_BUILD_DIR)
    message(FATAL_ERROR "Set LLAMA_CPP_DIR and LLAMA_BUILD_DIR to an existing checkout and build")
endif()
find_library(LLAMA_LIBRARY NAMES llama PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIBRARY NAMES ggml PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIBRARY NAMES ggml-base PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE LLAMA_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXECUTABLE}" --no-optional-locks -C "${LLAMA_CPP_DIR}" status --porcelain --untracked-files=no
    OUTPUT_VARIABLE source_diff OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(LLAMA_DIRTY "false")
if(source_diff)
    set(LLAMA_DIRTY "true")
endif()
set(LIBRARY_COMMIT "unknown")
if(EXISTS "${LLAMA_BUILD_DIR}/common/build-info.cpp")
    file(READ "${LLAMA_BUILD_DIR}/common/build-info.cpp" library_info)
    string(REGEX MATCH "LLAMA_COMMIT = \"([^\"]+)\"" unused "${library_info}")
    if(CMAKE_MATCH_1)
        set(LIBRARY_COMMIT "${CMAKE_MATCH_1}")
    endif()
endif()
file(SHA256 "${LLAMA_LIBRARY}" LIBRARY_SHA256)
file(SHA256 "${LLAMA_CPP_DIR}/include/llama.h" HEADER_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" WRAPPER_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/capture.hpp" CAPTURE_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS main.cpp capture.hpp
    "${LLAMA_CPP_DIR}/include/llama.h" "${LLAMA_LIBRARY}")
configure_file(build_info.hpp.in build_info.hpp @ONLY)

# Import only: never add_subdirectory(llama.cpp) or build any upstream target.
add_executable(glm53_nextn_probe main.cpp)
target_compile_features(glm53_nextn_probe PRIVATE cxx_std_17)
target_include_directories(glm53_nextn_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
    "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/ggml/include" "${LLAMA_CPP_DIR}/vendor")
target_link_libraries(glm53_nextn_probe PRIVATE "${LLAMA_LIBRARY}" "${GGML_LIBRARY}"
    "${GGML_BASE_LIBRARY}" ${CMAKE_DL_LIBS})
set_target_properties(glm53_nextn_probe PROPERTIES BUILD_RPATH "${LLAMA_BUILD_DIR}/bin")

enable_testing()
add_executable(nextn_capture_test test_capture.cpp)
target_compile_features(nextn_capture_test PRIVATE cxx_std_17)
target_include_directories(nextn_capture_test PRIVATE "${LLAMA_CPP_DIR}/vendor")
add_test(NAME nextn_capture_cpu COMMAND nextn_capture_test)
