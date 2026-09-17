// Pack a Kimodo LLM2Vec component directory into one seekable weight GGUF.
// Tensor payloads are streamed directly between files; even the BF16 model is
// never materialized in host memory as a complete model.
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
#include <unordered_set>
#include <vector>

namespace {

struct gguf_owner {
    gguf_context *file = nullptr;
    ggml_context *tensors = nullptr;
    ~gguf_owner() {
        if (file) gguf_free(file);
        if (tensors) ggml_free(tensors);
    }
};

struct output_owner {
    gguf_context *file = gguf_init_empty();
    ~output_owner() { if (file) gguf_free(file); }
};

struct tensor_source {
    std::filesystem::path path;
    std::uint64_t offset = 0;
    std::size_t bytes = 0;
};

void copy_bytes(std::ifstream &input, std::ofstream &output, std::uint64_t offset,
                std::size_t bytes, std::vector<char> &scratch) {
    input.seekg(static_cast<std::streamoff>(offset));
    for (std::size_t done = 0; done < bytes;) {
        const auto count = std::min(scratch.size(), bytes - done);
        input.read(scratch.data(), static_cast<std::streamsize>(count));
        if (!input) throw std::runtime_error("truncated tensor payload");
        output.write(scratch.data(), static_cast<std::streamsize>(count));
        if (!output) throw std::runtime_error("cannot write packed tensor payload");
        done += count;
    }
}

} // namespace

int main(int argc, char **argv) try {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " COMPONENT_BUNDLE OUTPUT.gguf\n";
        return 2;
    }
    const std::filesystem::path bundle = argv[1], output_path = argv[2];
    if (!std::filesystem::is_directory(bundle))
        throw std::runtime_error("component bundle is not a directory");
    if (std::filesystem::exists(output_path))
        throw std::runtime_error("output already exists: " + output_path.string());

    output_owner output;
    if (!output.file) throw std::runtime_error("cannot create output GGUF");
    std::vector<tensor_source> sources;
    std::unordered_set<std::string> names;
    bool copied_metadata = false;

    auto add_component = [&](const std::filesystem::path &path, int layer) {
        gguf_owner source;
        gguf_init_params params{true, &source.tensors};
        source.file = gguf_init_from_file(path.string().c_str(), params);
        if (!source.file || !source.tensors)
            throw std::runtime_error("cannot load component " + path.string());
        if (!copied_metadata) {
            gguf_set_kv(output.file, source.file);
            copied_metadata = true;
        }
        const auto data_offset = gguf_get_data_offset(source.file);
        for (std::int64_t index = 0; index < gguf_get_n_tensors(source.file); ++index) {
            const char *short_name = gguf_get_tensor_name(source.file, index);
            auto *tensor = ggml_get_tensor(source.tensors, short_name);
            if (!tensor) throw std::runtime_error("component tensor descriptor is missing");
            const std::string packed_name = layer >= 0
                ? "layer." + (layer < 10 ? std::string("0") : std::string()) +
                      std::to_string(layer) + "." + short_name
                : std::string(short_name);
            if (!names.insert(packed_name).second)
                throw std::runtime_error("duplicate packed tensor " + packed_name);
            ggml_set_name(tensor, packed_name.c_str());
            gguf_add_tensor(output.file, tensor);
            sources.push_back({path,
                static_cast<std::uint64_t>(data_offset + gguf_get_tensor_offset(source.file, index)),
                ggml_nbytes(tensor)});
        }
    };

    add_component(bundle / "embedding.gguf", -1);
    for (int layer = 0; layer < 32; ++layer) {
        std::array<char, 32> name{};
        std::snprintf(name.data(), name.size(), "layer-%02d.gguf", layer);
        add_component(bundle / name.data(), layer);
    }
    add_component(bundle / "final-norm.gguf", -1);

    gguf_set_val_str(output.file, "general.architecture", "kimodo-llm2vec");
    gguf_set_val_str(output.file, "kimodo.component", "text_encoder");
    gguf_set_val_u32(output.file, "kimodo.format_version", 2);
    gguf_set_val_u32(output.file, "kimodo.layer_count", 32);
    if (sources.empty() || sources.size() != names.size())
        throw std::runtime_error("packed tensor catalog is empty or inconsistent");

    if (!output_path.parent_path().empty())
        std::filesystem::create_directories(output_path.parent_path());
    const auto temporary = output_path.string() + ".tmp";
    struct temporary_cleanup {
        std::string path;
        ~temporary_cleanup() { std::error_code ignored; std::filesystem::remove(path, ignored); }
    } cleanup{temporary};
    if (!gguf_write_to_file(output.file, temporary.c_str(), true))
        throw std::runtime_error("cannot write packed GGUF metadata");

    std::ofstream packed(temporary, std::ios::binary | std::ios::app);
    if (!packed) throw std::runtime_error("cannot append packed GGUF data");
    const auto data_offset = static_cast<std::uint64_t>(gguf_get_meta_size(output.file));
    std::vector<char> scratch(8U*1024U*1024U);
    for (std::size_t index = 0; index < sources.size(); ++index) {
        const auto target = data_offset + gguf_get_tensor_offset(output.file, static_cast<std::int64_t>(index));
        const auto position = static_cast<std::uint64_t>(packed.tellp());
        if (position > target) throw std::runtime_error("packed tensor offsets overlap");
        if (position < target) {
            std::vector<char> padding(static_cast<std::size_t>(target-position));
            packed.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        }
        std::ifstream source(sources[index].path, std::ios::binary);
        if (!source) throw std::runtime_error("cannot reopen " + sources[index].path.string());
        copy_bytes(source, packed, sources[index].offset, sources[index].bytes, scratch);
    }
    const auto alignment = static_cast<std::uint64_t>(gguf_get_alignment(output.file));
    const auto written = static_cast<std::uint64_t>(packed.tellp());
    const auto padding = (alignment - written % alignment) % alignment;
    if (padding) {
        std::vector<char> zeros(static_cast<std::size_t>(padding));
        packed.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    packed.close();
    if (!packed) throw std::runtime_error("cannot finalize packed GGUF");
    std::filesystem::rename(temporary, output_path);
    cleanup.path.clear();
    std::cout << "wrote " << output_path << " (" << sources.size() << " tensors, "
              << std::filesystem::file_size(output_path) << " bytes)\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "text packing failed: " << error.what() << '\n';
    return 1;
}
