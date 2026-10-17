"""Shared descriptor checks for metadata-only and real tensor binding."""

import torch


def validate_bindings(model, module, descriptors):
    lazy = (
        module.Linear,
        module.ParallelEmbedding,
        module.ParallelEngramEmbedding,
        module.ParallelHead,
    )
    used = set()
    dense_types = {"BF16", "F32", "F16"}

    def require(name, shape, dtypes):
        if name not in descriptors:
            raise KeyError("Missing checkpoint tensor: " + name)
        desc = descriptors[name]
        if tuple(desc["shape"]) != tuple(shape) or desc["dtype"] not in dtypes:
            raise ValueError(
                f"{name}: got {desc}; expected shape={tuple(shape)}, dtype in {sorted(dtypes)}"
            )
        used.add(name)
        return desc

    for name, obj in model.named_modules():
        if isinstance(obj, lazy):
            key = name + ".weight"
            expected_dtype = obj.weight.dtype
            if expected_dtype == torch.float4_e2m1fn_x2:
                allowed = {"I8", "U8", "F4"}
            elif expected_dtype == torch.float8_e4m3fn:
                allowed = {"F8_E4M3"}
            else:
                allowed = dense_types | (
                    {"F8_E4M3"} if name.endswith("wo_a") else set()
                )
            desc = require(key, obj.weight.shape, allowed)
            shape, dtype = desc["shape"], desc["dtype"]
            if dtype in ("F8_E4M3", "I8", "U8", "F4"):
                if dtype in ("I8", "U8", "F4"):
                    scale_shape = (shape[0], shape[1] // 16)
                elif isinstance(obj, module.ParallelEngramEmbedding):
                    scale_shape = (shape[0], shape[1] // 32)
                else:
                    scale_shape = ((shape[0] + 31) // 32, (shape[1] + 31) // 32)
                require(name + ".scale", scale_shape, {"F8_E8M0"})
            continue
        for key, param in obj.named_parameters(recurse=False):
            full = name + "." + key if name else key
            require(full, param.shape, dense_types)
    unexpected = [
        name
        for name in descriptors
        if name.startswith("layers.")
        and name not in used
        and not name.endswith(".gate.bias_vl")
    ]
    if unexpected:
        raise ValueError("Unbound backbone tensors: " + ", ".join(unexpected[:10]))
    return used
