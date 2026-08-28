#include "internal.hpp"

#include <fstream>
#include <iostream>

namespace {
template<class T> std::vector<T> read(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const auto bytes=stream.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes)%sizeof(T)!=0U) throw std::runtime_error("invalid input size");
    std::vector<T> output(static_cast<std::size_t>(bytes)/sizeof(T));
    stream.seekg(0); stream.read(reinterpret_cast<char *>(output.data()), bytes); return output;
}
skintokens::device_kind device(std::string_view value) {
    return value=="cpu" ? skintokens::device_kind::cpu : skintokens::device_kind::vulkan;
}
}

int main(int argc, char ** argv) try {
    if (argc != 6) {
        std::cerr << "usage: skintokens-logits-dump MODEL_DIR CONDITION.f32 TOKENS.i32 cpu|vulkan OUTPUT.f32\n";
        return 2;
    }
    skintokens::runtime_options options; options.device=device(argv[4]);
    auto backend=skintokens::detail::make_backend(options);
    if (!backend) throw std::runtime_error(backend.error().message);
    auto weights=skintokens::detail::load_component(std::filesystem::path{argv[1]}/"tokenrig.gguf",(*backend)->value);
    if (!weights) throw std::runtime_error(weights.error().message);
    const auto condition=read<float>(argv[2]);
    const auto tokens=std::string_view{argv[3]}=="start" ? std::vector<std::int32_t>{257,266} : read<std::int32_t>(argv[3]);
    auto logits=skintokens::detail::run_qwen_logits(**weights,(*backend)->value,condition,tokens);
    if (!logits) throw std::runtime_error(logits.error().message);
    std::ofstream output(argv[5],std::ios::binary|std::ios::trunc);
    output.write(reinterpret_cast<const char *>(logits->data()),static_cast<std::streamsize>(logits->size()*sizeof(float)));
    std::cout << "wrote " << logits->size() << " logits\n";
    return output ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << exception.what() << '\n'; return 1;
}
