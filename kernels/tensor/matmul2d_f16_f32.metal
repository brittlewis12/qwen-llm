#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;

// One 16 x 16 output tile with a 32-wide reduction. The host selector only
// chooses this witness for exactly this shape; the portable kernel handles
// all other dimensions and all devices without Metal 4 support.
kernel void kernel_tensor_matmul2d_f16_f32(
        device half * left [[buffer(0)]],
        device half * right [[buffer(1)]],
        device float * output [[buffer(2)]],
        uint3 group [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(16, 16, 32, false, true);
    tensor_ops::matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<device half, extents<int, 32, 16>, tensor_inline> a(
        left + group.y * 16 * 32, extents<int, 32, 16>());
    tensor<device half, extents<int, 32, 16>, tensor_inline> b(
        right + group.x * 16 * 32, extents<int, 32, 16>());
    tensor<device float, extents<int, 16, 16>, tensor_inline> c(
        output + group.y * 16 * 16 + group.x * 16,
        extents<int, 16, 16>());
    auto destination = operation.get_destination_cooperative_tensor<
        decltype(a), decltype(b), float>();
    operation.run(a, b, destination);
    destination.store(c);
}
