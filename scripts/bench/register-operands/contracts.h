#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace probe_contracts {
inline uint64_t add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::runtime_error("price overflow");
    return a + b;
}
inline uint64_t pages(uint64_t bytes, uint64_t page) {
    if (!page || (page & (page - 1))) throw std::runtime_error("unknown page size");
    return bytes ? add(bytes, page - 1) / page * page : 0;
}
struct Domains {
    uint64_t physical, cpu_available, mwss, current_metal, max_buffer, page;
    uint64_t free_pages = 0, inactive_pages = 0, anonymous_pages = 0;
};
struct Price {
    std::array<uint64_t, 7> metal_lengths;
    uint64_t metal = 0, cpu_vectors = 0, cpu_touched = 0, cpu_control = 0;
    uint64_t cpu() const { return add(add(cpu_vectors, cpu_touched), cpu_control); }
    uint64_t incremental() const { return add(metal, cpu()); }
};
inline const char *admit(const Domains &d, const Price &p) {
    if (!d.physical || !d.mwss || !d.max_buffer || !d.page || (d.page & (d.page - 1))
        || d.cpu_available > d.physical) return "unknown/inconsistent telemetry";
    for (auto length : p.metal_lengths) if (length > d.max_buffer) return "maxBufferLength";
    if (d.current_metal > d.mwss || p.metal > d.mwss - d.current_metal) return "Metal MWSS headroom";
    if (p.cpu() > d.cpu_available) return "CPU/private availability";
    if (d.current_metal > d.physical || p.incremental() > d.physical - d.current_metal)
        return "combined owned Metal plus CPU physical capacity";
    return nullptr;
}
inline int slot_step(int count) {
    if (count < 1 || count > 64) throw std::runtime_error("unsupported count");
    int step = 17;
    while (std::gcd(step, count) != 1) ++step;
    return step;
}
inline int fixture_slot(int expert, int index, int count, bool dense, bool invalid_controls, int limit) {
    if (index < 0 || index >= count) throw std::runtime_error("slot index out of range");
    if (invalid_controls && !dense && index == 3) return -1;
    if (invalid_controls && !dense && index == 7) return limit;
    return dense ? index : expert * (count + 2) + index * slot_step(count) % count;
}
}
