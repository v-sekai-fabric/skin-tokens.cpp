add_executable(skintokens-api-test api_test.cpp)
target_link_libraries(skintokens-api-test PRIVATE skintokens::skintokens)
add_test(NAME skintokens-api-test COMMAND skintokens-api-test)

add_executable(skintokens-c-api-test c_api_test.c)
target_link_libraries(skintokens-c-api-test PRIVATE skintokens::skintokens)
add_test(NAME skintokens-c-api-test COMMAND skintokens-c-api-test)

# Compile the exact production integration source into a small fixture test so
# final learned-joint selection is covered without downloading model weights.
add_executable(skintokens-binding-test binding_test.cpp ../src/binding.cpp)
target_include_directories(skintokens-binding-test PRIVATE ../src ../include)
add_test(NAME skintokens-binding-test COMMAND skintokens-binding-test)

add_executable(skintokens-tokenizer-test tokenizer_test.cpp ../src/tokenizer.cpp)
target_include_directories(skintokens-tokenizer-test PRIVATE ../src ../include)
target_link_libraries(skintokens-tokenizer-test PRIVATE ggml)
add_test(NAME skintokens-tokenizer-test COMMAND skintokens-tokenizer-test)
