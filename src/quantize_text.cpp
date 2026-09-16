// Quantize the large BF16 matrices in a Kimodo LLM2Vec component bundle.
// Norms and the supervised F32 LoRA branch deliberately retain their source
// types so quantization experiments change only the dominant base weights.
#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct source_component {
    gguf_context *gguf = nullptr;
    ggml_context *ggml = nullptr;
    ~source_component() {
        if (gguf)
            gguf_free(gguf);
        if (ggml)
            ggml_free(ggml);
    }
};

struct output_context {
    gguf_context *value = gguf_init_empty();
    ~output_context() {
        if (value)
            gguf_free(value);
    }
};

struct quantization_profile {
    ggml_type default_type;
    bool mixed;
};

quantization_profile parse_profile(std::string_view name) {
    if (name == "q8_0")
        return {GGML_TYPE_Q8_0, false};
    if (name == "q6_k")
        return {GGML_TYPE_Q6_K, false};
    if (name == "q5_k")
        return {GGML_TYPE_Q5_K, false};
    if (name == "q4_k")
        return {GGML_TYPE_Q4_K, false};
    if (name == "q4_k_m")
        return {GGML_TYPE_Q4_K, true};
    throw std::runtime_error(
        "quantization must be one of q8_0, q6_k, q5_k, q4_k, q4_k_m");
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

bool should_quantize(const ggml_tensor &tensor) {
    const std::string_view name(tensor.name);
    return tensor.type == GGML_TYPE_BF16 && ggml_n_dims(&tensor) == 2 &&
           (name == "token_embedding.weight" ||
            ends_with(name, "_base.weight"));
}

ggml_type tensor_type(const ggml_tensor &tensor, quantization_profile profile) {
    if (!profile.mixed)
        return profile.default_type;
    const std::string_view name(tensor.name);
    if (name == "token_embedding.weight" ||
        name.find("attn_q_proj") != name.npos ||
        name.find("attn_k_proj") != name.npos)
        return GGML_TYPE_Q5_K;
    if (name.find("attn_v_proj") != name.npos ||
        name.find("ffn_down_proj") != name.npos)
        return GGML_TYPE_Q6_K;
    return GGML_TYPE_Q4_K;
}

std::vector<std::uint8_t> read_bytes(std::ifstream &input, std::uint64_t offset,
                                     std::size_t size) {
    std::vector<std::uint8_t> result(size);
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char *>(result.data()),
               static_cast<std::streamsize>(result.size()));
    if (!input)
        throw std::runtime_error("truncated tensor data");
    return result;
}

void quantize_component(const std::filesystem::path &input_path,
                        const std::filesystem::path &output_path,
                        quantization_profile profile,
                        std::string_view quant_name) {
    source_component source;
    gguf_init_params params{true, &source.ggml};
    source.gguf = gguf_init_from_file(input_path.string().c_str(), params);
    if (!source.gguf || !source.ggml)
        throw std::runtime_error("cannot load " + input_path.string());
    if (gguf_find_key(source.gguf, "kimodo.quantization") >= 0)
        throw std::runtime_error("refusing to requantize " +
                                 input_path.string());

    output_context output;
    if (!output.value)
        throw std::runtime_error("cannot allocate output GGUF context");
    gguf_set_kv(output.value, source.gguf);
    gguf_set_val_str(output.value, "kimodo.quantization",
                     std::string(quant_name).c_str());
    gguf_set_val_str(
        output.value, "kimodo.quantization_recipe",
        profile.mixed
            ? "Q4_K base; Q5_K token/Q/K; Q6_K V/FFN-down; BF16 norms; F32 "
              "supervised LoRA"
            : "uniform selected base matrices; BF16 norms; F32 supervised "
              "LoRA");
    gguf_set_val_u32(output.value, "general.quantization_version",
                     GGML_QNT_VERSION);

    const auto count = gguf_get_n_tensors(source.gguf);
    std::vector<std::vector<std::uint8_t>> payloads;
    payloads.reserve(static_cast<std::size_t>(count));
    std::ifstream input(input_path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot reopen " + input_path.string());
    const std::uint64_t data_offset = gguf_get_data_offset(source.gguf);
    std::size_t source_bytes = 0, output_bytes = 0, quantized_tensors = 0;

    for (std::int64_t index = 0; index < count; ++index) {
        const char *name = gguf_get_tensor_name(source.gguf, index);
        auto *tensor = ggml_get_tensor(source.ggml, name);
        if (!tensor)
            throw std::runtime_error("missing tensor descriptor " +
                                     std::string(name));
        gguf_add_tensor(output.value, tensor);
        const std::size_t bytes = ggml_nbytes(tensor);
        source_bytes += bytes;
        auto raw = read_bytes(
            input, data_offset + gguf_get_tensor_offset(source.gguf, index),
            bytes);

        if (should_quantize(*tensor)) {
            const ggml_type quant_type = tensor_type(*tensor, profile);
            const std::int64_t row_width = tensor->ne[0];
            const std::int64_t rows = ggml_nrows(tensor);
            if (row_width % ggml_blck_size(quant_type) != 0)
                throw std::runtime_error(std::string(name) +
                                         " row width is incompatible with " +
                                         std::string(quant_name));
            std::vector<float> values(
                static_cast<std::size_t>(ggml_nelements(tensor)));
            ggml_bf16_to_fp32_row(
                reinterpret_cast<const ggml_bf16_t *>(raw.data()),
                values.data(), static_cast<std::int64_t>(values.size()));
            payloads.emplace_back(ggml_row_size(quant_type, row_width) *
                                  static_cast<std::size_t>(rows));
            const std::size_t written = ggml_quantize_chunk(
                quant_type, values.data(), payloads.back().data(), 0, rows,
                row_width, nullptr);
            if (written != payloads.back().size())
                throw std::runtime_error("unexpected quantized size for " +
                                         std::string(name));
            gguf_set_tensor_type(output.value, name, quant_type);
            ++quantized_tensors;
        } else {
            payloads.push_back(std::move(raw));
        }
        gguf_set_tensor_data(output.value, name, payloads.back().data());
        output_bytes += payloads.back().size();
    }

    const auto temporary = output_path.string() + ".tmp";
    if (!gguf_write_to_file(output.value, temporary.c_str(), false))
        throw std::runtime_error("cannot write " + temporary);
    std::filesystem::rename(temporary, output_path);
    std::cout << input_path.filename().string() << ": " << quantized_tensors
              << " matrices, " << source_bytes << " -> " << output_bytes
              << " tensor bytes\n";
}

} // namespace

int main(int argc, char **argv) try {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " INPUT_BUNDLE OUTPUT_BUNDLE q8_0|q6_k|q5_k|q4_k|q4_k_m\n";
        return 2;
    }
    const std::filesystem::path input = argv[1], output = argv[2];
    const std::string quant_name = argv[3];
    const quantization_profile profile = parse_profile(quant_name);
    if (!std::filesystem::is_directory(input))
        throw std::runtime_error("input bundle is not a directory");
    const std::filesystem::path staging = output.string() + ".tmp";
    if (std::filesystem::exists(output) || std::filesystem::exists(staging))
        throw std::runtime_error("output bundle or staging directory already exists");
    std::filesystem::create_directories(staging);
    try {
        std::filesystem::copy_file(input / "tokenizer.gguf",
                                   staging / "tokenizer.gguf");
        quantize_component(input / "embedding.gguf", staging / "embedding.gguf",
                           profile, quant_name);
        quantize_component(input / "final-norm.gguf", staging / "final-norm.gguf",
                           profile, quant_name);
        for (int index = 0; index < 32; ++index) {
            std::array<char, 32> name{};
            std::snprintf(name.data(), name.size(), "layer-%02d.gguf", index);
            quantize_component(input / name.data(), staging / name.data(), profile,
                               quant_name);
        }
        std::filesystem::rename(staging, output);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
        throw;
    }
    std::cout << "wrote " << output << " (" << quant_name << ")\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "text quantization failed: " << error.what() << '\n';
    return 1;
}
