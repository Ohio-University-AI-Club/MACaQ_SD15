"""Runtime for the SD 1.5 MACaQ attention and feed-forward projections."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

VENDOR = Path(__file__).resolve().parent / "macaq_vendor"
sys.path.insert(0, str(VENDOR))
from macaq_vllm_mxfp4_dense import dequantize_mxfp4


class W4A4Projection(nn.Module):
    def __init__(self, in_features, tensors, backend):
        super().__init__()
        self.in_features = in_features
        self.padded_width = tensors["pre_scale"].numel()
        self.out_features = tensors["W_packed"].shape[0]
        self.backend = backend
        if backend == "native":
            from kernels import FusedW4A4Linear

            self.native = FusedW4A4Linear(
                tensors["W_packed"], tensors["W_scales"], tensors["B_inv_T"],
                tensors.get("bias"), 32, self.out_features, self.padded_width,
                pre_scale=tensors["pre_scale"],
            )
        else:
            for key in ("pre_scale", "B_inv_T", "bias"):
                self.register_buffer(key, tensors.get(key))
            self.register_buffer("weight_bf16", torch.empty(
                (self.out_features, self.padded_width), device=self.pre_scale.device, dtype=torch.bfloat16
            ))
            # Bound the workspace used to unpack a weight matrix.
            for start in range(0, self.out_features, 256):
                rows = slice(start, start + 256)
                self.weight_bf16[rows].copy_(dequantize_mxfp4(
                    tensors["W_packed"][rows], tensors["W_scales"][rows], self.padded_width
                ))

    def forward(self, x):
        if self.padded_width != self.in_features:
            x = F.pad(x, (0, self.padded_width - self.in_features))
        if self.backend == "native":
            return self.native(x)
        return self.emulate(x)

    def emulate(self, x):
        # Reproduce MACaQ's runtime activation rounding with a BF16 matmul.
        table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
        thresholds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
        out = []
        for rows in x.reshape(-1, self.padded_width).split(512):
            blocks = (rows.bfloat16() * self.pre_scale).reshape(len(rows), -1, 32)
            rotated = (blocks @ self.B_inv_T).float()
            scale = (rotated.abs().amax(-1, keepdim=True).clamp_min(1e-20).log2().floor() - 2)
            scale = scale.clamp(-127, 127).exp2()
            indices = torch.bucketize((rotated.abs() / scale).contiguous(), thresholds, right=True)
            activations = (table[indices].copysign(rotated) * scale).reshape_as(rows).bfloat16()
            value = F.linear(activations, self.weight_bf16)
            if self.bias is not None:
                value = (value.float() + self.bias.float()).bfloat16()
            out.append(value)
        return torch.cat(out).reshape(*x.shape[:-1], self.out_features).to(x.dtype)


def set_projection(unet, name, module):
    parent, _, child = name.rpartition(".")
    setattr(unet.get_submodule(parent), child, module.eval())


def load_packed_projections(unet, checkpoint_dir, backend="native"):
    """Attach the saved projection delta to the matching original SD 1.5 U-Net."""
    if backend not in {"native", "emulated"}:
        raise ValueError("backend must be native or emulated")
    checkpoint_dir = Path(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    if manifest["format"] != "sd15-macaq-projections-v1":
        raise ValueError("Unrecognized projection format")
    device = str(next(unet.parameters()).device)
    if backend == "native" and (not device.startswith("cuda") or torch.cuda.get_device_capability(device)[0] < 10):
        raise RuntimeError("Native MXFP4 requires a compatible Blackwell-class GPU")
    packed = load_file(str(checkpoint_dir / "projections.safetensors"))
    for name, spec in manifest["layers"].items():
        current = unet.get_submodule(name)
        if (current.in_features, current.out_features) != (spec["in_features"], spec["out_features"]):
            raise ValueError(f"Projection shape mismatch for {name}")
        prefix = name + "."
        keys = ["W_packed", "W_scales", "B_inv_T", "pre_scale"]
        if spec["bias"]:
            keys.append("bias")
        tensors = {key: packed.pop(prefix + key).to(device) for key in keys}
        set_projection(unet, name, W4A4Projection(spec["in_features"], tensors, backend))
    if packed:
        raise ValueError("Unrecognized saved projection tensors")
    return manifest
