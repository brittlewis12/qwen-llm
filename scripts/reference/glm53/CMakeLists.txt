cmake_minimum_required(VERSION 3.20)
project(glm53_oracle LANGUAGES C CXX)

# Upstream 11fe02151 plus a LoRA-only glm5-next attention wiring fix that does
# not execute without adapters. glm5-next is unchanged from 42d958167.
set(GLM53_REFERENCE_REVISION "e845373ff0b2b0998d673282bec05ee2f4e20106")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to the pinned llama.cpp checkout")
endif()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE reference_head OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" status --porcelain --untracked-files=no
    OUTPUT_VARIABLE reference_dirty OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT reference_head STREQUAL GLM53_REFERENCE_REVISION OR NOT reference_dirty STREQUAL "")
    message(FATAL_ERROR "Oracle requires the clean pinned llama.cpp revision ${GLM53_REFERENCE_REVISION}")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama")
add_executable(glm53_oracle main.cpp)
target_compile_features(glm53_oracle PRIVATE cxx_std_17)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" GLM53_WRAPPER_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" GLM53_CMAKE_SHA256)
target_compile_definitions(glm53_oracle PRIVATE
    GLM53_REFERENCE_REVISION="${GLM53_REFERENCE_REVISION}"
    GLM53_WRAPPER_SHA256="${GLM53_WRAPPER_SHA256}"
    GLM53_CMAKE_SHA256="${GLM53_CMAKE_SHA256}")
target_link_libraries(glm53_oracle PRIVATE llama)
set_target_properties(glm53_oracle PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
