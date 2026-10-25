#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include "contracts.h"

struct Args {
    uint32_t M, N, K, nb01, stride_b, min_count, max_count, b_div, slot_limit;
};
static constexpr size_t max_fixture_arms = 14;
static constexpr size_t max_name_capacity = 192;
static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void emit(NSDictionary *record) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil];
    fwrite(data.bytes, 1, data.length, stdout);
    puts("");
    fflush(stdout);
}
static uint32_t rng(uint32_t &state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static probe_contracts::Domains domains(id<MTLDevice> device) {
    uint64_t physical = 0;
    size_t length = sizeof(physical);
    require(sysctlbyname("hw.memsize", &physical, &length, nullptr, 0) == 0
            && length == sizeof(physical) && physical > 0, "physical telemetry unavailable");
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t size = HOST_VM_INFO64_COUNT;
    const mach_port_t host = mach_host_self();
    require(host != MACH_PORT_NULL, "host telemetry port unavailable");
    vm_size_t page = 0;
    const auto page_result = host_page_size(host, &page);
    const auto result = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &size);
    const auto released = mach_port_deallocate(mach_task_self(), host);
    const auto needed = (offsetof(vm_statistics64_data_t, internal_page_count) + sizeof(vm.internal_page_count)) / sizeof(integer_t);
    require(result == KERN_SUCCESS && size >= needed && size <= HOST_VM_INFO64_COUNT
            && page_result == KERN_SUCCESS && page > 0 && !(page & (page - 1)) && released == KERN_SUCCESS,
            "CPU telemetry unavailable/incomplete");
    using probe_contracts::add;
    for (uint64_t pages : {uint64_t(vm.free_count), uint64_t(vm.inactive_count), uint64_t(vm.internal_page_count)})
        require(pages <= physical / page, "inconsistent CPU telemetry");
    require(add(vm.free_count, vm.inactive_count) <= physical / page, "inconsistent free/inactive telemetry");
    const uint64_t available_pages = add(vm.free_count, vm.inactive_count > vm.internal_page_count
            ? vm.inactive_count - vm.internal_page_count : 0);
    require(available_pages <= physical / page, "inconsistent CPU availability");
    return {physical, available_pages * page, device.recommendedMaxWorkingSetSize,
            device.currentAllocatedSize, device.maxBufferLength, page,
            vm.free_count, vm.inactive_count, vm.internal_page_count};
}
static id<MTLBuffer> alloc(id<MTLDevice> device, uint64_t bytes) {
    require(bytes <= device.maxBufferLength, "buffer exceeds maxBufferLength");
    id<MTLBuffer> result = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(result != nil, "ordinary shared-buffer allocation failed");
    return result;
}

struct Fixture {
    Args args;
    int experts, count, block_bytes;
    bool dense, half_act;
    uint64_t weight_bytes, activation_bytes, output_bytes, id_bytes, audit_bytes, priced_bytes;
    id<MTLBuffer> w, x, counts, ids, out, audit, decoded;
    std::vector<float> seed;
    std::vector<uint8_t> touched;
    float activation(uint64_t element) const {
        return half_act ? float(((_Float16 *)x.contents)[element]) : ((float *)x.contents)[element];
    }
    Fixture(id<MTLDevice> device, int m, int k, int e, int c, int n, int bytes,
            bool parity, bool is_dense = false, bool f16 = false, bool tile32 = false)
            : experts(e), count(c), block_bytes(bytes), dense(is_dense), half_act(f16) {
        args = {uint32_t(m), uint32_t(n), uint32_t(k), uint32_t(k / 256 * bytes), uint32_t(k),
                0, UINT32_MAX / 2, 1, uint32_t(e * (c + 2) + 2)};
        if (dense) args.slot_limit = c;
        weight_bytes = uint64_t(e) * m * args.nb01;
        activation_bytes = uint64_t(args.slot_limit) * k * (half_act ? 2 : 4);
        output_bytes = uint64_t(args.slot_limit) * m * 4 + 128;
        id_bytes = uint64_t(e) * n * 4;
        audit_bytes = parity ? uint64_t(k / 32) * (tile32 ? 2048 : 6144) * 2 + (tile32 ? 16 : 0) : 4;
        const uint64_t decode_bytes = parity ? uint64_t(e) * m * k * 2 : 4;
        const auto d = domains(device);
        probe_contracts::Price price{{weight_bytes, activation_bytes, uint64_t(e) * 4, id_bytes,
                                      output_bytes, audit_bytes, decode_bytes}};
        NSMutableArray *metal_parts = [NSMutableArray new];
        for (uint64_t bytes : price.metal_lengths) {
            const uint64_t rounded = probe_contracts::pages(bytes, d.page);
            price.metal = probe_contracts::add(price.metal, rounded);
            [metal_parts addObject:@{@"logical_bytes": @(bytes), @"page_ceil_bytes": @(rounded)}];
        }
        // Three independent float arrays: seed, CPU reference, incumbent copy.
        price.cpu_vectors = parity ? 3 * probe_contracts::pages(output_bytes, d.page) : 0;
        price.cpu_touched = parity ? probe_contracts::pages(output_bytes / 4, d.page) : 0;
        const uint64_t cpu_control_payload = sizeof(Fixture) + 2 * sizeof(std::vector<float>)
            + sizeof(Args) + sizeof(probe_contracts::Price) + sizeof(probe_contracts::Domains)
            + (2 * max_fixture_arms + 4) * (sizeof(std::string) + max_name_capacity + 1);
        price.cpu_control = probe_contracts::pages(cpu_control_payload, d.page);
        priced_bytes = price.incremental();
        emit(@{@"event": @"price", @"parity": @(parity), @"M": @(m), @"K": @(k),
               @"experts": @(e), @"count": @(c), @"grid_N": @(n), @"weight_bytes": @(weight_bytes),
               @"activation_storage": half_act ? @"f16" : @"f32", @"activation_bytes": @(activation_bytes),
               @"price_schema": @"split-domains-v2", @"priced_bytes": @(priced_bytes),
               @"metal_buffers_in_binding_order": metal_parts, @"metal_incremental_page_bytes": @(price.metal),
               @"cpu_vector_page_bytes": @(price.cpu_vectors), @"cpu_touched_page_bytes": @(price.cpu_touched),
               @"cpu_control_page_bytes": @(price.cpu_control), @"cpu_incremental_page_bytes": @(price.cpu()),
               @"cpu_control_payload_bytes": @(cpu_control_payload), @"control_arm_bound": @(max_fixture_arms),
               @"control_name_capacity_bound": @(max_name_capacity),
               @"cpu_conservative_available_bytes": @(d.cpu_available), @"physical_capacity_bytes": @(d.physical),
               @"cpu_free_pages": @(d.free_pages), @"cpu_inactive_pages": @(d.inactive_pages),
               @"cpu_anonymous_pages": @(d.anonymous_pages),
               @"recommended_working_set_bytes": @(d.mwss), @"current_allocated_metal_bytes": @(d.current_metal),
               @"max_buffer_length": @(d.max_buffer), @"page_size": @(d.page),
               @"scope": @"incremental fixture buffers and explicit CPU arrays/controls; mapped Shared Metal counted once",
               @"unpriced": @"Foundation/Metal runtime objects, PSO/driver heaps and allocator metadata; no byte bound, reservation or total-RSS claim",
               @"framework_scope": @"two libraries, one queue, selected-arm PSO cache retained; one outstanding CB and a drained pool per command; fixture pool per case"});
        require(!parity || priced_bytes <= 128ull * 1024 * 1024, "tiny parity exceeds explicit 128 MiB bound");
        if (const char *denied = probe_contracts::admit(d, price)) throw std::runtime_error(denied);
        w = alloc(device, weight_bytes); x = alloc(device, activation_bytes);
        counts = alloc(device, e * 4); ids = alloc(device, id_bytes);
        out = alloc(device, output_bytes); audit = alloc(device, audit_bytes); decoded = alloc(device, decode_bytes);
        const auto start = std::chrono::steady_clock::now();
        uint32_t state = 0x987abcd;
        auto wb = (uint8_t *)w.contents;
        for (uint64_t block = 0; block < weight_bytes / bytes; ++block) {
            _Float16 scale = _Float16((1 + rng(state) % 8) / 4096.0f);
            memcpy(wb + block * bytes, &scale, 2);
            for (int j = 2; j < bytes; ++j) wb[block * bytes + j] = uint8_t(rng(state));
        }
        for (uint64_t i = 0; i < uint64_t(args.slot_limit) * k; ++i) {
            const float value = float(int(rng(state) % 20001) - 10000) / 8192.0f;
            if (half_act) ((_Float16 *)x.contents)[i] = _Float16(value);
            else ((float *)x.contents)[i] = value;
        }
        std::fill_n((int *)ids.contents, id_bytes / 4, -123456);
        if (parity) touched.resize(output_bytes / 4, false);
        for (int expert = 0; expert < e; ++expert) {
            ((int *)counts.contents)[expert] = c;
            for (int i = 0; i < c; ++i) {
                int slot = probe_contracts::fixture_slot(expert, i, c, dense, parity, int(args.slot_limit));
                ((int *)ids.contents)[expert * n + i] = slot;
                if (parity && slot >= 0 && slot < int(args.slot_limit))
                    for (int row = 0; row < m; ++row) touched[16 + slot * m + row] = true;
            }
        }
        // Count-filtered and empty experts test byte preservation independently
        // of invalid IDs. Never change counts in the matched timing fixtures.
        if (parity && !dense && e > 2) {
            args.max_count = c;
            ((int *)counts.contents)[e - 1] = 0;
            ((int *)counts.contents)[e - 2] = c + 1;
            ((int *)counts.contents)[e - 3] = -1;
            for (int slot = (e - 3) * (c + 2); slot < int(args.slot_limit); ++slot)
                for (int row = 0; row < m; ++row) touched[16 + slot * m + row] = false;
        }
        if (parity) {
            seed.resize(output_bytes / 4);
            for (size_t i = 0; i < seed.size(); ++i) seed[i] = float(int(i % 41) - 20) / 4;
        }
        memset(out.contents, 0xa5, output_bytes);
        emit(@{@"event": @"fixture_allocated", @"current_allocated_metal_bytes": @(device.currentAllocatedSize),
               @"admission_current_metal_bytes": @(d.current_metal),
               @"actual_seed_capacity_bytes": @(seed.capacity() * sizeof(float)),
               @"actual_touched_capacity_bytes": @(touched.capacity())});
        emit(@{@"event": @"fixture_preparation", @"seconds": @(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count()), @"packing": @"synthetic native packed blocks; no predecode for streaming",
            @"activation_preparation": half_act ? @"generated directly into rounded F16 storage; no F32 shadow or per-dispatch conversion"
                                                : @"generated F32 storage, half cast inside legacy kernels"});
    }
};

struct Probe {
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLLibrary> baseline_library;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    Probe(const char *path, const char *baseline_path) {
        if (@available(macOS 27.0, *)) {} else throw std::runtime_error("standalone probe requires macOS 27");
        device = MTLCreateSystemDefaultDevice();
        require(device != nil && [device supportsFamily:MTLGPUFamilyApple10], "probe requires M5 / Apple10");
        NSError *error = nil;
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
        if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
        baseline_library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(baseline_path)] error:&error];
        if (!baseline_library) throw std::runtime_error(error.localizedDescription.UTF8String);
        queue = [device newCommandQueue]; pipelines = [NSMutableDictionary new];
        emit(@{@"event": @"platform", @"device": device.name,
               @"os": NSProcessInfo.processInfo.operatingSystemVersionString, @"library_path": @(path),
               @"baseline_library_path": @(baseline_path),
               @"occupancy": @"not exposed by public PSO API; max threads is a limit, not occupancy",
               @"spills": @"not exposed by public PSO API"});
    }
    id<MTLComputePipelineState> pipeline(const std::string &name) {
        NSString *key = @(name.c_str());
        if (pipelines[key]) return pipelines[key];
        const bool same27 = name.starts_with("baseline27:");
        const std::string function = same27 ? name.substr(11) : name;
        id<MTLLibrary> selected = !same27 && name.starts_with("kernel_") ? baseline_library : library;
        id<MTLFunction> fn = [selected newFunctionWithName:@(function.c_str())];
        require(fn != nil, name.c_str());
        NSError *error = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) throw std::runtime_error(error.localizedDescription.UTF8String);
        pipelines[key] = pso;
        emit(@{@"event": @"pipeline", @"name": key, @"max_threads": @(pso.maxTotalThreadsPerThreadgroup),
               @"execution_width": @(pso.threadExecutionWidth), @"static_threadgroup_bytes": @(pso.staticThreadgroupMemoryLength)});
        return pso;
    }
    double run(Fixture &f, const std::string &name, int repeats = 1, bool dense_baseline = false) {
        @autoreleasepool {
        const bool single = name.starts_with("sg1");
        const bool tile32 = name.starts_with("sg1t32");
        require(f.half_act == (tile32 || name.find("_f16act_") != std::string::npos),
                "activation storage does not match pipeline signature");
        const bool direct = single && name.find("tg") == std::string::npos;
        const uint64_t threadgroup_bytes = direct ? 0 : tile32 ? 4096 : 8192;
        const uint64_t row_tile = tile32 ? 32 : 64;
        id<MTLComputePipelineState> pso = pipeline(name);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        const auto start = std::chrono::steady_clock::now();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBytes:&f.args length:sizeof(Args) atIndex:0];
        [enc setBuffer:f.w offset:0 atIndex:1]; [enc setBuffer:f.x offset:0 atIndex:2];
        if (dense_baseline) [enc setBuffer:f.out offset:64 atIndex:3];
        else {
            [enc setBuffer:f.counts offset:0 atIndex:3]; [enc setBuffer:f.ids offset:0 atIndex:4];
            [enc setBuffer:f.out offset:64 atIndex:5];
            float limit = 1.25f;
            [enc setBytes:&limit length:4 atIndex:6];
            [enc setBuffer:f.audit offset:0 atIndex:7];
        }
        if (threadgroup_bytes) [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
        for (int i = 0; i < repeats; ++i)
            [enc dispatchThreadgroups:MTLSizeMake((f.args.N + 31) / 32, (f.args.M + row_tile - 1) / row_tile, f.experts)
                threadsPerThreadgroup:MTLSizeMake(single ? 32 : 128, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error(cb.error.localizedDescription.UTF8String);
        const double gpu_ns = (cb.GPUEndTime - cb.GPUStartTime) * 1e9 / repeats;
        require(std::isfinite(gpu_ns) && gpu_ns > 0, "invalid GPU timestamp");
        emit(@{@"event": @"sample", @"name": @(name.c_str()), @"count": @(f.count),
               @"experts": @(f.experts), @"M": @(f.args.M), @"K": @(f.args.K),
               @"repeats": @(repeats), @"gpu_ns": @(gpu_ns),
               @"threads_per_group": @(single ? 32 : 128), @"dynamic_threadgroup_bytes": @(threadgroup_bytes),
               @"row_tile": @(row_tile), @"activation_storage": f.half_act ? @"f16" : @"f32",
               @"output_path": direct ? @"direct cooperative accumulator" : @"retained threadgroup output",
               @"host_ns_per_dispatch": @(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e9 / repeats)});
        return gpu_ns;
        }
    }
    void decode(Fixture &f, const std::string &dtype) {
        @autoreleasepool {
        Args a = f.args; a.M *= f.experts;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline("decode_" + dtype)];
        [enc setBytes:&a length:sizeof(a) atIndex:0]; [enc setBuffer:f.w offset:0 atIndex:1];
        [enc setBuffer:f.decoded offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(uint64_t(a.M) * a.K / 16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        require(cb.status == MTLCommandBufferStatusCompleted, "decode command failed");
        emit(@{@"event": @"parity_predecode", @"gpu_ns": @((cb.GPUEndTime - cb.GPUStartTime) * 1e9),
               @"bytes": @(f.decoded.length), @"timing_path_uses_predecode": @NO});
        }
    }
};

static std::string baseline(const std::string &dtype, int epi, bool dense = false, bool half_act = false) {
    const std::string infix = half_act ? "_f16act" : "_f32";
    if (dense) return "kernel_mat_mat_" + dtype + infix + "_tensor";
    return std::string(epi == 0 ? "kernel_moe_down_" : epi == 1 ? "kernel_moe_up_silu_mul_" : "kernel_moe_up_clamped_")
        + dtype + infix + "_grouped_slots_tensor";
}
static void audit(Fixture &f, bool weight_left) {
    const auto got = (_Float16 *)f.audit.contents, w = (_Float16 *)f.decoded.contents;
    uint64_t mismatches = 0, checked = 0;
    for (uint32_t k = 0; k < f.args.K; k += 32) for (int sg = 0; sg < 4; ++sg)
        for (int side = 0; side < 2; ++side) {
            const bool is_weight = weight_left ? side == 0 : side == 1;
            const int base = is_weight ? (weight_left ? sg * 16 : (sg % 2) * 32)
                                       : (weight_left ? 0 : (sg / 2) * 16);
            for (int row = 0; row < (side == 0 ? 16 : 32); ++row) for (int q = 0; q < 32; ++q) {
                _Float16 expected;
                if (is_weight) expected = w[std::min(base + row, int(f.args.M) - 1) * f.args.K + k + q];
                else {
                    int slot = ((int *)f.ids.contents)[std::min(base + row, f.count - 1)];
                    if (slot < 0 || slot >= int(f.args.slot_limit)) slot = 0;
                    expected = _Float16(f.activation((slot / f.args.b_div) * f.args.K + k + q));
                }
                auto value = got[(uint64_t(k / 32) * 4 + sg) * 1536 + (side ? 512 : 0) + row * 32 + q];
                mismatches += memcmp(&expected, &value, 2) != 0; ++checked;
            }
        }
    emit(@{@"event": @"operand_parity", @"mapping": weight_left ? @"wx" : @"ax",
           @"half_values_checked": @(checked), @"bit_mismatches": @(mismatches)});
    require(mismatches == 0, "decoded-half cooperative operand mismatch");
}
static void audit_single(Fixture &f, bool tile32 = false) {
    const auto got = (_Float16 *)f.audit.contents + (tile32 ? 8 : 0), w = (_Float16 *)f.decoded.contents;
    const int weight_rows = tile32 ? 32 : 64;
    if (tile32) {
        auto traits = (uint32_t *)f.audit.contents;
        emit(@{@"event": @"cooperative_traits", @"left_half_per_lane": @(traits[0]),
               @"right_half_per_lane": @(traits[1]), @"acc_float_per_lane": @(traits[2]), @"simd_width": @(traits[3])});
        require(traits[0] == 32 && traits[1] == 32 && traits[2] == 32 && traits[3] == 32,
                "32x32 cooperative trait mismatch");
    }
    uint64_t mismatches = 0, checked = 0;
    uint64_t weight_errors = 0, activation_errors = 0;
    NSMutableArray *first_differences = [NSMutableArray new];
    for (uint32_t k = 0; k < f.args.K; k += 32) for (int row = 0; row < weight_rows + 32; ++row)
        for (int q = 0; q < 32; ++q) {
            _Float16 expected;
            if (row < weight_rows) expected = w[std::min(row, int(f.args.M) - 1) * f.args.K + k + q];
            else {
                int slot = ((int *)f.ids.contents)[std::min(row - weight_rows, f.count - 1)];
                if (slot < 0 || slot >= int(f.args.slot_limit)) slot = 0;
                expected = _Float16(f.activation((slot / f.args.b_div) * f.args.K + k + q));
            }
            const auto value = got[uint64_t(k / 32) * (weight_rows + 32) * 32 + row * 32 + q];
            if (memcmp(&expected, &value, 2) != 0) {
                ++mismatches;
                if (row < weight_rows) ++weight_errors; else ++activation_errors;
                if (first_differences.count < 32) {
                    uint16_t expected_bits, got_bits;
                    memcpy(&expected_bits, &expected, 2); memcpy(&got_bits, &value, 2);
                    [first_differences addObject:@{@"k": @(k), @"row": @(row), @"q": @(q),
                        @"expected_bits": @(expected_bits), @"got_bits": @(got_bits)}];
                }
            }
            ++checked;
        }
    emit(@{@"event": @"operand_parity", @"mapping": tile32 ? @"single SIMD 32x32 F16ACT" : @"single SIMD whole macro",
           @"half_values_checked": @(checked), @"bit_mismatches": @(mismatches),
           @"weight_errors": @(weight_errors), @"activation_errors": @(activation_errors),
           @"first_differences": first_differences});
    require(mismatches == 0, "single SIMD cooperative operand mismatch");
}
static void parity(Probe &p, const std::string &dtype, int count, int k, int epi, bool dense, bool resident = false,
                   int b_div = 1, bool single = false, bool tile32 = false) {
    const uint64_t before = p.device.currentAllocatedSize;
    @autoreleasepool {
    Fixture f(p.device, 70, k, dense ? 1 : 8, count, dense ? count : 64,
              dtype == "iq3_xxs" ? 98 : 66, true, dense, tile32, tile32);
    f.args.b_div = b_div;
    p.decode(f, dtype);
    std::vector<float> ref = f.seed;
    const auto w = (_Float16 *)f.decoded.contents;
    for (int e = 0; e < f.experts; ++e) {
        const int count_e = ((int *)f.counts.contents)[e];
        if (count_e < int(f.args.min_count) || count_e > int(f.args.max_count)) continue;
        for (int j = 0; j < count_e; ++j) {
        int slot = ((int *)f.ids.contents)[e * f.args.N + j];
        if (slot < 0 || slot >= int(f.args.slot_limit)) continue;
        for (uint32_t row = 0; row < f.args.M; ++row) {
            double sum = 0;
            for (int q = 0; q < k; ++q) {
                const int kk = resident ? q % 32 : q;
                sum += double(w[(e * f.args.M + row) * k + kk]) * double(_Float16(f.activation((slot / f.args.b_div) * k + kk)));
            }
            float value = float(sum), gate = ref[16 + slot * f.args.M + row];
            if (epi) {
                if (epi == 2) { gate = std::min(gate, 1.25f); value = std::clamp(value, -1.25f, 1.25f); }
                value *= gate / (1.0f + std::exp(-gate));
            }
            ref[16 + slot * f.args.M + row] = value;
        }
        }
    }
    std::vector<float> base;
    std::vector<std::string> mappings = {"baseline", "ax", "wx"};
    if (tile32) {
        mappings = {"baseline", "baseline27", "sg1t32", "sg1t32tg"};
        if (epi == 0) { mappings.push_back("sg1t32_audit"); mappings.push_back("sg1t32tg_audit"); }
    } else if (single) {
        mappings = {"baseline", "baseline27", "sg1wx512", "sg1wx512tg", "sg1ax512", "sg1ax512tg", "sg1axbt16", "sg1axbt16tg"};
        if (epi == 0) for (const std::string name : {"sg1wx512", "sg1wx512tg", "sg1ax512", "sg1ax512tg", "sg1axbt16", "sg1axbt16tg"})
            mappings.push_back(name + "_audit");
    } else if (!resident) mappings.insert(mappings.begin() + 1, "baseline27");
    if (epi == 0 && !resident && !single && !tile32) {
        mappings.push_back("ax16"); mappings.push_back("wx16");
        mappings.push_back("axsh"); mappings.push_back("wxsh");
        for (const std::string name : {"ax", "wx", "ax16", "wx16", "axsh", "wxsh"})
            mappings.push_back(name + "_audit");
    }
    for (const std::string &mapping : mappings) {
        require(mappings.size() <= max_fixture_arms && mappings.capacity() <= 2 * max_fixture_arms
                && mapping.capacity() <= max_name_capacity, "control collection exceeds priced bound");
        memcpy(f.out.contents, f.seed.data(), f.output_bytes);
        memset(f.audit.contents, 0xa5, f.audit_bytes);
        const bool is_audit = mapping.ends_with("_audit");
        const std::string operand_mapping = is_audit ? mapping.substr(0, mapping.size() - 6) : mapping;
        std::string name = mapping == "baseline27" ? "baseline27:" + baseline(dtype, epi, dense, f.half_act)
            : resident ? (mapping == "baseline" ? "staged" : mapping) + "_resident_" + dtype
            : mapping == "baseline" ? baseline(dtype, epi, dense, f.half_act)
            : operand_mapping + (epi == 0 ? (is_audit ? "_audit_" : "_") : epi == 1 ? "_silu_" : "_clamp_") + dtype;
        require(name.capacity() <= max_name_capacity && operand_mapping.capacity() <= max_name_capacity,
                "control name exceeds priced bound");
        p.run(f, name, 1, (mapping == "baseline" || mapping == "baseline27") && dense);
        auto got = (float *)f.out.contents;
        double max_abs = 0, ref_max = 0, base_abs = 0, base_max = 0, dot = 0, ng = 0, nc = 0;
        uint64_t bits = 0, untouched = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!f.touched[i]) {
                require(memcmp(got + i, f.seed.data() + i, 4) == 0, "untouched tail/guard/invalid ID byte changed");
                ++untouched; continue;
            }
            require(std::isfinite(got[i]), "non-finite parity output");
            max_abs = std::max(max_abs, std::abs(double(got[i]) - ref[i]));
            ref_max = std::max(ref_max, std::abs(double(ref[i])));
            dot += double(got[i]) * ref[i]; ng += double(got[i]) * got[i]; nc += double(ref[i]) * ref[i];
            if (!base.empty()) {
                bits += memcmp(got + i, base.data() + i, 4) != 0;
                base_abs = std::max(base_abs, std::abs(double(got[i]) - base[i]));
                base_max = std::max(base_max, std::abs(double(base[i])));
            }
        }
        double cosine = dot / std::max(1e-30, std::sqrt(ng * nc));
        emit(@{@"event": @"output_parity", @"dtype": @(dtype.c_str()), @"mapping": @(mapping.c_str()),
               @"count": @(count), @"K": @(k), @"epi": @(epi), @"dense": @(dense),
               @"resident": @(resident), @"b_div": @(b_div),
               @"activation_storage": f.half_act ? @"f16" : @"f32",
               @"max_abs_vs_half_reference": @(max_abs), @"reference_max": @(ref_max), @"cosine": @(cosine),
               @"bit_mismatches_vs_baseline": @(bits), @"max_abs_vs_baseline": @(base_abs),
               @"untouched_float_words": @(untouched), @"command_status": @"completed"});
        require(cosine > .999 && max_abs <= 1e-2 * ref_max + 1e-4, "existing numerical reference contract failed");
        require(base.empty() || base_abs <= 2e-3 * base_max + 1e-5, "existing tensor comparison contract failed");
        if (mapping == "baseline") base.assign(got, got + ref.size());
        else if (is_audit) {
            if (single || tile32) audit_single(f, tile32);
            else audit(f, mapping.starts_with("wx"));
        }
    }
    emit(@{@"event": @"cpu_array_capacities", @"seed_bytes": @(f.seed.capacity() * sizeof(float)),
           @"reference_bytes": @(ref.capacity() * sizeof(float)), @"baseline_bytes": @(base.capacity() * sizeof(float)),
           @"touched_bytes": @(f.touched.capacity())});
    }
    emit(@{@"event": @"fixture_scope_end", @"count": @(count), @"K": @(k),
           @"before_fixture_metal_bytes": @(before), @"after_pool_metal_bytes": @(p.device.currentAllocatedSize),
           @"retained_pipeline_count": @(p.pipelines.count),
           @"scope": @"fixture C++ destruction and autorelease pool drained; PSO cache retained; no baseline tolerance asserted"});
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        try {
            require(argc >= 5, "usage: probe /absolute/private.metallib /absolute/baseline26.metallib parity|timing|parity-single|timing-single|parity-32x32|timing-32x32 iq3_xxs|iq2_xxs [count] [samples]");
            require(argv[1][0] == '/' && argv[2][0] == '/', "explicit absolute metallib paths required");
            const std::string mode = argv[3], dtype = argv[4];
            require(dtype == "iq3_xxs" || dtype == "iq2_xxs", "unsupported dtype");
            Probe p(argv[1], argv[2]);
            const bool single = mode == "parity-single" || mode == "timing-single";
            const bool tile32 = mode == "parity-32x32" || mode == "timing-32x32";
            require(!(single || tile32) || dtype == "iq3_xxs", "new shape experiments initially IQ3_XXS only");
            if (mode == "parity-32x32") {
                parity(p, dtype, 32, 512, 0, false, false, 1, false, true);
                parity(p, dtype, 32, 2304, 0, false, false, 1, false, true);
                parity(p, dtype, 32, 2304, 0, true, false, 1, false, true);
                for (int epi : {1, 2}) parity(p, dtype, 32, 512, epi, false, false, 6, false, true);
            } else if (mode == "parity-single") {
                for (int count : {1, 32, 33}) parity(p, dtype, count, 512, 0, false, false, 1, true);
                parity(p, dtype, 32, 2304, 0, true, false, 1, true);
                parity(p, dtype, 32, 2304, 0, false, false, 1, true);
                for (int epi : {1, 2}) parity(p, dtype, 32, 512, epi, false, false, 6, true);
            } else if (mode == "parity") {
                for (int count : {1, 2, 8, 16, 31, 32, 33}) for (int epi : {0, 1, 2})
                    parity(p, dtype, count, 512, epi, false);
                for (int count : {32, 64}) parity(p, dtype, count, 2304, 0, true);
                parity(p, dtype, 33, 2304, 0, false);
                parity(p, dtype, 32, 512, 0, false, true);
                parity(p, dtype, 16, 512, 2, false, false, 6);
            } else {
                require(mode == "timing" || mode == "timing-single" || mode == "timing-32x32", "unknown mode");
                int count = argc > 5 ? std::stoi(argv[5]) : 32;
                int samples = argc > 6 ? std::stoi(argv[6]) : 9;
                require(count >= 1 && count <= 64 && samples >= 2 && samples <= 31, "invalid bounded timing parameters");
                const uint64_t before = p.device.currentAllocatedSize;
                @autoreleasepool {
                Fixture f(p.device, 5120, 2304, 384, count, 2048, dtype == "iq3_xxs" ? 98 : 66, false, false, tile32, tile32);
                const double useful = 2.0 * f.args.M * f.args.K * f.experts * count;
                const double padded = 2.0 * f.args.M * f.args.K * f.experts * ((count + 31) / 32 * 32);
                if (tile32) emit(@{@"event": @"work", @"useful_flops": @(useful), @"padded_flops": @(padded),
                       @"packed_bank_bytes": @(f.weight_bytes), @"activation_storage": @"f16",
                       @"incumbent_row_tile": @64, @"candidate_row_tile": @32, @"K_step": @32,
                       @"incumbent_operand_staging_write_bytes": @(padded / 131072 * 6144),
                       @"candidate_operand_staging_write_bytes": @0,
                       @"incumbent_logical_activation_read_bytes": @(padded / 131072 * 2048),
                       @"candidate_logical_activation_read_bytes": @(padded / 131072 * 4096),
                       @"candidate_retained_output_staging_write_bytes": @(padded / (2.0 * f.args.K * 32 * 32) * 4096),
                       @"candidate_direct_output_staging_bytes": @0,
                       @"traffic_caveat": @"logical bytes, not DRAM; changed M/32 grid and output ownership; identical stored half activations"});
                else emit(@{@"event": @"work", @"useful_flops": @(useful), @"padded_flops": @(padded),
                       @"packed_bank_bytes": @(f.weight_bytes),
                       @"unique_packed_bytes_per_active_panel": @(f.weight_bytes * ((count + 31) / 32)),
                       @"baseline_operand_staging_write_bytes": @(padded / 131072 * 6144),
                       @"cooperative_operand_staging_bytes": @0,
                       @"baseline_logical_activation_read_bytes": @(padded / 131072 * 4096),
                       @"ax_logical_activation_read_bytes": @(padded / 131072 * 8192),
                       @"wx_logical_activation_read_bytes": @(padded / 131072 * 16384),
                       @"retained_output_staging_write_bytes": @(padded / (2.0 * f.args.K * 32 * 64) * 8192),
                       @"traffic_caveat": @"logical bytes, not measured DRAM; retained-output controls and direct-output variants differ; duplicate SIMD loads differ"});
                std::vector<std::string> names = {baseline(dtype, 0), "baseline27:" + baseline(dtype, 0), "ax_" + dtype, "wx_" + dtype,
                    "ax16_" + dtype, "wx16_" + dtype,
                    "axsh_" + dtype, "wxsh_" + dtype,
                    "staged_resident_" + dtype, "ax_resident_" + dtype, "wx_resident_" + dtype};
                if (tile32) names = {baseline(dtype, 0, false, true), "baseline27:" + baseline(dtype, 0, false, true),
                    "sg1t32_" + dtype, "sg1t32tg_" + dtype};
                else if (single) names = {baseline(dtype, 0), "baseline27:" + baseline(dtype, 0), "wx16_" + dtype,
                    "sg1wx512_" + dtype, "sg1wx512tg_" + dtype, "sg1ax512_" + dtype, "sg1ax512tg_" + dtype,
                    "sg1axbt16_" + dtype, "sg1axbt16tg_" + dtype};
                require(names.size() <= max_fixture_arms && names.capacity() <= 2 * max_fixture_arms,
                        "timing controls exceed priced bound");
                if (tile32) emit(@{@"event": @"ownership_variant", @"single_simd": @YES, @"shape": @"32x32x32",
                       @"scope": @"one accumulator; M/32 grid versus incumbent M/64; matched F16ACT storage; direct/retained output controls"});
                else emit(@{@"event": @"ownership_variant", @"single_simd": @(single),
                       @"scope": @"same 64x32 macro and full grid; one SIMD, four compatible 512-point accumulators with input reuse; direct/retained output are separate variants",
                       @"single_simd_logical_activation_read_bytes": @(padded / 131072 * 4096),
                       @"single_simd_direct_output_staging_bytes": @0});
                for (const auto &name : names) { p.pipeline(name); p.run(f, name); }
                for (int i = 0; i < samples; ++i) for (int j = 0; j < int(names.size()); ++j)
                    p.run(f, names[(j + i) % names.size()], 3);
                double checksum = 0;
                for (uint64_t i = 16; i < f.output_bytes / 4 - 16; i += 997) checksum += ((float *)f.out.contents)[i];
                emit(@{@"event": @"consumed_output", @"checksum": @(checksum)});
                }
                emit(@{@"event": @"fixture_scope_end", @"before_fixture_metal_bytes": @(before),
                       @"after_pool_metal_bytes": @(p.device.currentAllocatedSize),
                       @"retained_pipeline_count": @(p.pipelines.count),
                       @"scope": @"timing fixture and command pools drained; PSO cache retained; no release tolerance asserted"});
            }
            return 0;
        } catch (const std::exception &error) {
            fprintf(stderr, "probe: %s\n", error.what()); return 1;
        }
    }
}
