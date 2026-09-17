#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>

namespace kimodo::detail {

// A cached graph owns both its GGML metadata context and a dedicated compute
// allocator. Dedicated allocators keep tensor addresses stable and avoid
// re-running GGML's graph allocation pass when switching between stages.
struct cached_motion_graph {
    ~cached_motion_graph() {
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
    }

    ggml_context *context = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_gallocr *allocator = nullptr;
    ggml_tensor *output = nullptr;
    std::vector<ggml_tensor *> inputs;
    std::vector<std::vector<float>> retained_inputs;
    std::size_t output_values = 0;
    std::size_t scratch_bytes = 0;
};

struct cached_motion_transformer {
    std::string prefix;
    std::size_t motion_dim = 0;
    std::vector<std::unique_ptr<cached_motion_graph>> stages;
    std::vector<float> last_embedding;
    std::vector<float> last_headings;
};

// Motion requests commonly repeat one shape for tens or hundreds of diffusion
// steps. Retain only that most-recent shape so interactive changes do not grow
// memory without bound.
struct motion_graph_cache {
    std::size_t batch = 0;
    std::size_t frames = 0;
    int layer_chunk = 0;
    std::vector<std::unique_ptr<cached_motion_transformer>> transformers;
};

} // namespace kimodo::detail
