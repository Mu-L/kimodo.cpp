#pragma once

#include <chrono>
#include <cstdlib>

namespace kimodo::detail {

inline bool profile_enabled() noexcept {
    return std::getenv("KIMODO_PROFILE") != nullptr;
}

inline double profile_elapsed_ms(std::chrono::steady_clock::time_point started) noexcept {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

} // namespace kimodo::detail
