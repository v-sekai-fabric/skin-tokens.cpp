cmake_minimum_required(VERSION 3.25)
project(skin-tokens.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SKINTOKENS_BUILD_TESTS "Build tests" ON)
option(SKINTOKENS_ENABLE_VULKAN "Build GGML Vulkan backend" ON)
option(SKINTOKENS_DYNAMIC_BACKENDS "Load CPU/Vulkan backends dynamically" ON)
option(SKINTOKENS_CPU_ALL_VARIANTS "Build portable CPU backend variants" ON)
option(SKINTOKENS_SANITIZE "Enable ASan and UBSan" OFF)
option(SKINTOKENS_FUZZ "Build parser fuzzers" OFF)
set(SKINTOKENS_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH "Pinned GGML submodule")

if(SKINTOKENS_SANITIZE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "SKINTOKENS_SANITIZE requires Clang")
  endif()
  add_library(skintokens_sanitizers INTERFACE)
  target_compile_options(skintokens_sanitizers INTERFACE -fno-omit-frame-pointer -fsanitize=address,undefined)
  target_link_options(skintokens_sanitizers INTERFACE -fno-omit-frame-pointer -fsanitize=address,undefined)
endif()

if(NOT EXISTS "${SKINTOKENS_GGML_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Initialize the ggml git submodule or set SKINTOKENS_GGML_SOURCE_DIR")
endif()
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${SKINTOKENS_ENABLE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ${SKINTOKENS_DYNAMIC_BACKENDS} CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ${SKINTOKENS_CPU_ALL_VARIANTS} CACHE BOOL "" FORCE)
if(SKINTOKENS_DYNAMIC_BACKENDS)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()
add_subdirectory("${SKINTOKENS_GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)

add_library(skintokens SHARED
  src/backend.cpp
  src/binding.cpp
  src/c_api.cpp
  src/gguf.cpp
  src/glb.cpp
  src/mesh_encoder.cpp
  src/model.cpp
  src/qwen.cpp
  src/skinning.cpp
  src/skin_vae.cpp
  src/tokenizer.cpp
  src/weights.cpp
)
add_library(skintokens::skintokens ALIAS skintokens)
target_include_directories(skintokens PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> PRIVATE src)
target_compile_definitions(skintokens PRIVATE SKINTOKENS_BUILD SKINTOKENS_SHARED)
target_compile_definitions(skintokens PUBLIC SKINTOKENS_SHARED)
target_link_libraries(skintokens PRIVATE ggml Threads::Threads nlohmann_json::nlohmann_json)
if(TARGET skintokens_sanitizers)
  target_link_libraries(skintokens PUBLIC skintokens_sanitizers)
endif()
set_target_properties(skintokens PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
if(MSVC)
  target_compile_options(skintokens PRIVATE /W4 /permissive- /sdl)
else()
  target_compile_options(skintokens PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wformat=2)
endif()

add_executable(skintokens-cli src/cli.cpp)
target_link_libraries(skintokens-cli PRIVATE skintokens::skintokens)
set_target_properties(skintokens-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-qwen-parity
  tools/qwen_parity.cpp src/backend.cpp src/qwen.cpp src/weights.cpp)
target_include_directories(skintokens-qwen-parity PRIVATE src include)
target_link_libraries(skintokens-qwen-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-qwen-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-mesh-parity
  tools/mesh_parity.cpp src/backend.cpp src/mesh_encoder.cpp src/weights.cpp)
target_include_directories(skintokens-mesh-parity PRIVATE src include)
target_link_libraries(skintokens-mesh-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-mesh-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-vae-parity
  tools/vae_parity.cpp src/backend.cpp src/skin_vae.cpp src/weights.cpp)
target_include_directories(skintokens-vae-parity PRIVATE src include)
target_link_libraries(skintokens-vae-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-vae-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-generation-parity
  tools/generation_parity.cpp src/backend.cpp src/qwen.cpp src/weights.cpp)
target_include_directories(skintokens-generation-parity PRIVATE src include)
target_link_libraries(skintokens-generation-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-generation-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-animation-parity tools/animation_parity.cpp)
target_link_libraries(skintokens-animation-parity PRIVATE skintokens::skintokens nlohmann_json::nlohmann_json)
set_target_properties(skintokens-animation-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-full-binding-parity
  tools/full_binding_parity.cpp src/backend.cpp src/binding.cpp src/skin_vae.cpp src/weights.cpp)
target_include_directories(skintokens-full-binding-parity PRIVATE src include)
target_link_libraries(skintokens-full-binding-parity PRIVATE ggml Threads::Threads nlohmann_json::nlohmann_json)
set_target_properties(skintokens-full-binding-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-logits-dump
  tools/logits_dump.cpp src/backend.cpp src/qwen.cpp src/weights.cpp)
target_include_directories(skintokens-logits-dump PRIVATE src include)
target_link_libraries(skintokens-logits-dump PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-logits-dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(CTest)
if(SKINTOKENS_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(SKINTOKENS_FUZZ)
  add_subdirectory(fuzz)
endif()
