"""Generate standalone counterfactuals from the maintained K32 tensor body."""

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[3]
s = (root / "kernels/tensor/moe_grouped_slots_tensor.metal").read_text()
a = s.index("template <int BYTES, short NL,")
b = s.index("// 64-slot gate/up tile.", a)
body = s[a:b]
body = body.replace(
    "short EPI, bool DENSE, bool HALF_ACT>",
    "short EPI, bool DENSE, bool HALF_ACT, int PANEL, int HOLD>",
)
body = body.replace("moe_grouped_slots_mm_tensor_body", "probe_body")
body = body.replace("TT_NR1", "PANEL")
# Generalize activation/epilogue ownership while retaining the original weight mapping.
start = body.index("    const short pr1 =")
end = body.index("    const short offset1", start)
body = body[:start] + body[end:]
start = body.index("    const int slot_id =")
end = body.index("    threadgroup half * stage_a", start)
body = body[:start] + body[end:]
body = body.replace("    threadgroup half * stage_b = sb + pr1 * TT_NK + iy;\n", "")
start = body.index("    if (!slot_valid)")
end = body.index("    // left =", start)
body = body[:start] + body[end:]
body = body.replace(
    "        DEQ(x_ptr, il, temp_a);",
    """        if (HOLD == 2) {
            device const half * p = decoded + ((ulong(im) * args.M + r0 + lr0) * args.K + loop_k + 16 * il0);
            for (short j = 0; j < 4; ++j) temp_a[j] = *((device const half4 *)(p + 4*j));
        } else if (HOLD == 3) {
            for (short j = 0; j < 4; ++j) temp_a[j] = half4(0.0625h);
        } else { DEQ(x_ptr, il, temp_a); }""",
)
start = body.index("        const half2x4 yv =")
end = body.index("        // The previous", start)
body = body[:start] + body[end:]
start = body.index("        if (slot_valid)")
end = body.index("        il =", start)
body = (
    body[:start]
    + """        for (uint a = tiitg; a < PANEL * 4; a += 128) {
            const int j = int(a / 4), iy = 8 * int(a % 4);
            const int slot = ids[ulong(im) * args.N + r1 + min(j, int(nr1)-1)];
            const bool valid = j < nr1 && slot >= 0 && slot < slot_limit;
            const ulong off = ulong(valid ? slot / int(args.b_div) : 0) * args.stride_b + loop_k + iy;
            const half2x4 yv = !valid ? half2x4(0.0h) : HALF_ACT
                ? *((device const half2x4 *)srcB_bytes + off / 8)
                : half2x4(*((device const float2x4 *)srcB_bytes + off / 8));
            *((threadgroup half2x4 *)(sb + j * TT_NK + iy)) = yv;
        }

"""
    + body[end:]
)
body = body.replace("        x_ptr = (il < 2)", "        x_ptr = (HOLD != 4 && il < 2)")
body = body.replace("        y_ptr += TT_NK;\n        yh_ptr += TT_NK;\n", "")
# The no-MMA twin writes volatile shared operands so the compiler must retain
# every dequantized Half, without adding a reduction in the hot K loop.
for j in [0, 4, 8, 12]:
    old = f"            *(threadgroup half4 *)(stage_a + {j})  = temp_a[{j // 4}];"
    if j == 12:
        old = old.replace(")  =", ") =")
    body = body.replace(
        old,
        f"            if (HOLD == 1) *(volatile threadgroup half4 *)(stage_a + {j}) = temp_a[{j // 4}]; else "
        + old.strip(),
    )
old = "*((threadgroup half2x4 *)(sb + j * TT_NK + iy)) = yv;"
body = body.replace(
    old,
    "if (HOLD == 1) { *((volatile threadgroup half4 *)(sb+j*TT_NK+iy))=yv[0]; *((volatile threadgroup half4 *)(sb+j*TT_NK+iy+4))=yv[1]; } else "
    + old,
)
body = body.replace(
    "        op.run(left, right, acc);",
    "        if (HOLD != 1) op.run(left, right, acc);",
)
body = body.replace(
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    threadgroup float * temp_str",
    "    if (HOLD == 1) { sink[((ulong(tgpig.z) * ((args.M+63)/64) + tgpig.y) * ((args.N+PANEL-1)/PANEL) + tgpig.x)*128+tiitg] = float(*((volatile threadgroup half *)sa + tiitg * 16)); return; }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    threadgroup float * temp_str",
)
body = body.replace(
    "    const short j = tiitg / 4;",
    "    for (short j = tiitg / 4; j < PANEL; j += 32) {",
)
# extra brace for epilogue loop
body = body.rstrip()[:-1] + "    }\n}\n"
body = body.replace(
    "        float limit) {",
    "        float limit, device const half * decoded, device float * sink) {",
)
header = (
    '#include "'
    + str(root / "kernels/tensor/moe_grouped_slots_tensor.metal")
    + '"\n'
    + '#include "'
    + str(root / "scripts/bench/glm-tensor-experts/iq_fast_decoders.metalh")
    + '"\n'
)
# The generated absolute include is build output only, never committed.
out = header + body
formats = [
    ("iq2_s", 82, True),
    ("iq2_xs", 74, True),
    ("iq2_xxs", 66, True),
    ("iq3_s", 110, False),
    ("iq3_xxs", 98, True),
]
for dtype, block, half in formats:
    for panel in [16, 32, 64]:
        variants = [
            (f"probe_{dtype}_n{panel}_h{hold}", f"qt_dequantize_{dtype}", hold, "")
            for hold in range(5)
        ]
        variants.append(
            (f"probe_{dtype}_n{panel}_fast", f"qt_dequantize_{dtype}_fast", 0, "_fast")
        )
        if dtype.startswith("iq2_"):
            variants.append(
                (f"probe_{dtype}_n{panel}_vec", f"qt_dequantize_{dtype}_vec", 0, "_vec")
            )
        for name, deq, hold, suffix in variants:
            out += f"""\nkernel void {name}(
 constant moe_group_generic_mm_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
 device const uchar *x [[buffer(2)]], device const int *c [[buffer(3)]], device const int *ids [[buffer(4)]],
 device float *o [[buffer(5)]], constant float &limit [[buffer(6)]], device const half *decoded [[buffer(8)]],
 device float *sink [[buffer(9)]], threadgroup uchar *sm [[threadgroup(0)]],
 uint3 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {{
 probe_body<QT_{dtype.upper()}_BYTES, QT_{dtype.upper()}_NL, {deq}, 0, false, {str(half).lower()}, {panel}, {hold}>(a,w,x,c,ids,o,sm,grid,tid,limit,decoded,sink);
}}
"""
    unpack_suffixes = ["", "_fast"] + (["_vec"] if dtype.startswith("iq2_") else [])
    for suffix in unpack_suffixes:
        out += f"""\nkernel void unpack_{dtype}{suffix}(device const uchar *w [[buffer(0)]], device half *h [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
 half4x4 v; qt_dequantize_{dtype}{suffix}(w + ulong(i / 16) * QT_{dtype.upper()}_BYTES, short(i%16), v);
 for (short j=0;j<4;++j) *((device half4 *)(h+ulong(i)*16+4*j))=v[j];
}}
"""
Path(sys.argv[1]).write_text(out)
