if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "SkinTokens fuzzers require Clang")
endif()

add_executable(skintokens-glb-fuzzer glb_fuzzer.cpp)
target_link_libraries(skintokens-glb-fuzzer PRIVATE skintokens::skintokens)
target_compile_options(skintokens-glb-fuzzer PRIVATE -fsanitize=fuzzer,address,undefined)
target_link_options(skintokens-glb-fuzzer PRIVATE -fsanitize=fuzzer,address,undefined)
