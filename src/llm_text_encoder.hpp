#pragma once

#include <array>
#include <expected>
#include <memory>
#include <string>
#include <string_view>

namespace kimodo::detail {

class llm_text_encoder {
public:
    // A text model is either a monolithic weight GGUF beside tokenizer.gguf,
    // or a legacy component directory. Layers are loaded in bounded groups by
    // default. KIMODO_TEXT_LAYER_CHUNK=32 keeps the complete encoder resident
    // while executing bounded GGML graphs. KIMODO_TEXT_RESIDENT_LIMIT_MIB can
    // impose a VRAM-safe cap.
    static std::expected<std::unique_ptr<llm_text_encoder>, std::string> load(std::string_view source);
    std::expected<std::array<float, 4096>, std::string> encode(std::string_view utf8_prompt) const;
    ~llm_text_encoder();
    llm_text_encoder(const llm_text_encoder &) = delete;
    llm_text_encoder &operator=(const llm_text_encoder &) = delete;
private:
    llm_text_encoder() = default;
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace kimodo::detail
