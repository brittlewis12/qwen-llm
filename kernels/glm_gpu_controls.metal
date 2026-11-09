// Request-local sparse decode controls, ordered before the ordinary GLM body.
#include <metal_stdlib>
using namespace metal;
struct glm_gpu_controls { uint visible; uint pool; uint count; };
kernel void kernel_glm_gpu_controls(
    constant glm_gpu_controls &a [[buffer(0)]],
    device int *pools [[buffer(1)]], device int *rows [[buffer(2)]],
    device int *status [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) { pools[0] = int(a.visible/a.pool); rows[0] = int(a.visible); }
    if (i < a.count) status[i] = -1;
}
