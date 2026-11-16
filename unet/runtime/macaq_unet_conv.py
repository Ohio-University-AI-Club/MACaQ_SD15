"""SD 1.5 convolution adapter for the MACaQ dense MXFP4 runtime."""
import math
from pathlib import Path
import json

import torch
from torch import nn
from torch.nn import functional as F
from sd15_macaq_runtime import W4A4Projection, set_projection


def patch_view(x, kernel, stride, padding, dilation):
    ph, pw = padding
    kh, kw = kernel
    dh, dw = dilation
    if ph or pw:
        x = F.pad(x, (pw, pw, ph, ph))
    return x.unfold(2, dh * (kh - 1) + 1, stride[0]).unfold(
        3, dw * (kw - 1) + 1, stride[1]
    )[..., ::dh, ::dw].permute(0, 2, 3, 1, 4, 5)


def patch_rows(view, indices):
    _, h, w = view.shape[:3]
    return view[indices // (h * w), (indices // w) % h, indices % w].flatten(1)


class W4A4Conv2d(nn.Module):
    def __init__(self, spec, tensors, backend, tile_rows=256):
        super().__init__()
        self.spec = spec
        self.linear = W4A4Projection(spec["in_features"], tensors, backend)
        self.tile_rows = tile_rows

    def forward(self, x):
        s = self.spec
        view = patch_view(x, s["kernel"], s["stride"], s["padding"], s["dilation"])
        b, h, w = view.shape[:3]
        out = []
        for start in range(0, b * h * w, self.tile_rows):
            i = torch.arange(start, min(start + self.tile_rows, b * h * w), device=x.device)
            out.append(self.linear(patch_rows(view, i)))
        return torch.cat(out, 0).reshape(b, h, w, s["out_features"]).permute(0, 3, 1, 2).contiguous()


def build_replacement(original, spec, tensors, backend, tile_rows):
    if spec["kind"] == "conv2d":
        if not isinstance(original, nn.Conv2d) or list(original.weight.shape) != spec["weight_shape"]:
            raise ValueError("Source convolution does not match the packed layer")
        return W4A4Conv2d(spec, tensors, backend, tile_rows)
    if not isinstance(original, nn.Linear) or tuple(original.weight.shape) != (
        spec["out_features"], spec["in_features"]
    ):
        raise ValueError("Source linear does not match the packed layer")
    return W4A4Projection(spec["in_features"], tensors, backend)


def load_macaq_unet(directory, backend="native", device="cuda", tile_rows=256):
    """Load a complete exported U-Net with no source U-Net weights required."""
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file

    device = torch.device(device)
    if backend not in ("native", "emulated") or tile_rows < 1:
        raise ValueError("Invalid backend or tile_rows")
    if backend == "native" and (
        device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10
    ):
        raise RuntimeError("Native MXFP4 requires a Blackwell-class GPU")
    directory = Path(directory)
    info = json.loads((directory / "manifest.json").read_text())
    if info["format"] != "sd15-macaq-full-unet-conv-v1":
        raise ValueError("Unexpected U-Net checkpoint format")
    config = UNet2DConditionModel.load_config(directory)
    with init_empty_weights():
        model = UNet2DConditionModel.from_config(config)
    state = load_file(str(directory / "model.safetensors"))
    expected_missing = set()
    for name, spec in info["layers"].items():
        prefix = name + "."
        keys = ["W_packed", "W_scales", "pre_scale", "B_inv_T"]
        if spec["bias"]:
            keys.append("bias")
        tensors = {key: state.pop(prefix + key).to(device) for key in keys}
        packed = build_replacement(model.get_submodule(name), spec, tensors, backend, tile_rows)
        set_projection(model, name, packed)
        expected_missing.update(prefix + k for k in packed.state_dict())
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if set(missing) != expected_missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: {set(missing) - expected_missing}, {unexpected}")
    return model.eval().to(device)
