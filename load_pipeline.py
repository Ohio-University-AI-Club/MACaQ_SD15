"""Load the uploaded SD 1.5 MACaQ model and its bundled components."""
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionPipeline
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer


def load_pipeline(repo_or_directory, device="cuda", backend="native", token=None):
    directory = Path(repo_or_directory)
    if not directory.is_dir():
        directory = Path(snapshot_download(repo_or_directory, token=token))
    sys.path.insert(0, str(directory / "unet" / "runtime"))
    from macaq_unet_conv import load_macaq_unet

    unet = load_macaq_unet(directory / "unet", backend=backend, device=device)
    encoder_dir = directory / "text_encoder"
    if (encoder_dir / "manifest.json").is_file():
        sys.path.insert(0, str(encoder_dir / "runtime"))
        from load_macaq_clip import load_macaq_clip

        text_encoder = load_macaq_clip(encoder_dir, backend=backend, device=device)
    else:
        text_encoder = CLIPTextModel.from_pretrained(encoder_dir, torch_dtype=torch.float16).eval().to(device)
    vae = AutoencoderKL.from_pretrained(directory / "vae", torch_dtype=torch.float16).eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    scheduler = LCMScheduler.from_pretrained(directory / "scheduler", local_files_only=True)
    return StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    )
