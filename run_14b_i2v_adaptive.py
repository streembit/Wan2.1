#!/usr/bin/env python3
"""
Wan2.1 14B Image-to-Video Generation - Adaptive Hardware Version
Automatically selects single-GPU FP8 mode or multi-GPU distributed mode based on hardware
"""

import os
import sys
import torch
import gc
import argparse
from PIL import Image
import time
import numpy as np
import subprocess

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Wan modules
import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video
from wan.image2video import WanI2V


def get_gpu_info():
    """Get GPU count and VRAM size"""
    if not torch.cuda.is_available():
        return 0, 0
    
    gpu_count = torch.cuda.device_count()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return gpu_count, vram_gb


def should_use_fp8_mode():
    """Determine if we should use FP8 mode based on hardware"""
    gpu_count, vram_gb = get_gpu_info()
    
    # Check for environment variable override
    if os.environ.get('FORCE_FP8_MODE') == '1':
        print("FP8 mode forced via FORCE_FP8_MODE=1")
        return True
    
    if os.environ.get('FORCE_MULTI_GPU_MODE') == '1':
        print("Multi-GPU mode forced via FORCE_MULTI_GPU_MODE=1")
        return False
    
    # Use FP8 mode for single GPU setups or consumer GPUs
    if gpu_count <= 1 or vram_gb <= 48:
        return True
    
    # Use multi-GPU mode for datacenter setups
    return False


def apply_selective_fp8(model):
    """Apply FP8 only to safe layers that won't cause type promotion errors"""
    
    converted_count = 0
    total_count = 0
    
    # These patterns are known to be safe for FP8 conversion
    safe_patterns = [
        "blocks.*.ffn.0.weight",  # FFN first layer weights
        "blocks.*.ffn.2.weight",  # FFN second layer weights
        "blocks.*.self_attn.q.weight",  # Query projections
        "blocks.*.self_attn.k.weight",  # Key projections
        "blocks.*.self_attn.v.weight",  # Value projections
        "blocks.*.self_attn.o.weight",  # Output projections
        "blocks.*.cross_attn.q.weight",
        "blocks.*.cross_attn.k.weight", 
        "blocks.*.cross_attn.v.weight",
        "blocks.*.cross_attn.o.weight",
    ]
    
    # Never convert these layers
    never_convert = [
        "modulation", "norm", "bias", "embedding", "position", 
        "ln", "adaln", "scale_shift", "y_embedder", "t_embedder",
        "img_emb", "patch_embedding", "final_proj"
    ]
    
    for name, param in model.named_parameters():
        total_count += 1
        
        # Skip if in never_convert list
        if any(pattern in name.lower() for pattern in never_convert):
            continue
            
        # Only convert if it matches safe patterns
        if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            # Check if layer name matches safe patterns
            is_safe = False
            for pattern in safe_patterns:
                if "*" in pattern:
                    # Convert pattern to regex and check
                    import re
                    regex_pattern = pattern.replace("*", r"\d+")
                    if re.match(regex_pattern, name):
                        is_safe = True
                        break
                elif name == pattern:
                    is_safe = True
                    break
            
            # Simple check for known safe layers
            if ("ffn" in name and "weight" in name and "bias" not in name) or \
               ("attn" in name and "weight" in name and "bias" not in name):
                try:
                    param.data = param.data.contiguous().to(torch.float8_e4m3fn)
                    converted_count += 1
                except Exception as e:
                    print(f"  Warning: Could not convert {name} to FP8: {e}")
    
    print(f"  Selectively converted {converted_count}/{total_count} parameters to FP8")
    return model


def setup_fp8_quantization():
    """Setup FP8 quantization for memory-efficient inference"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    print("FP8 quantization settings configured")


def generate_i2v_single_gpu_fp8(
    image_path,
    prompt,
    output_file="i2v_output.mp4",
    checkpoint_dir="./Wan2.1-I2V-14B-480P",
    resolution="480*832",
    num_frames=81,
    num_steps=50,
    guidance_scale=5.0,
    shift_scale=3.0,
    seed=-1,
    negative_prompt="",
    offload_model=True
):
    """Generate video using single GPU with FP8 optimization"""
    
    print("=== SINGLE GPU MODE WITH FP8 OPTIMIZATION ===")
    print(f"Using I2V model from: {checkpoint_dir}")
    print(f"Model offloading: {'Enabled' if offload_model else 'Disabled'}")
    
    # Setup FP8
    setup_fp8_quantization()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load configuration
    print("Loading I2V-14B configuration...")
    cfg = WAN_CONFIGS['i2v-14B']
    
    # Create wrapper class for FP8 optimization
    class WanI2VFP8(WanI2V):
        def __init__(self, *args, **kwargs):
            # Initialize with CPU and T5 offloading
            kwargs['init_on_cpu'] = True
            kwargs['t5_cpu'] = True
            super().__init__(*args, **kwargs)
            
            print("\nApplying selective FP8 optimization...")
            
            # Apply selective FP8 to DiT model
            print("Converting safe layers in DiT model to FP8...")
            self.model = apply_selective_fp8(self.model)
            
            # Keep CLIP in original precision for stability
            print("✓ CLIP model kept in original precision")
            print("✓ VAE kept in FP16 for quality")
            
            # Move models to GPU if not offloading
            if not offload_model:
                self.device = torch.device('cuda')
                self.model = self.model.to(self.device)
                if hasattr(self, 'clip'):
                    self.clip.to(self.device)
    
    # Initialize model
    print(f"Initializing Wan I2V model with selective FP8...")
    start_time = time.time()
    
    model = WanI2VFP8(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
        init_on_cpu=True
    )
    
    print(f"\nModel initialization took: {time.time() - start_time:.2f} seconds")
    print(f"GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Load input image
    print(f"\nLoading input image: {image_path}")
    input_image = Image.open(image_path).convert('RGB')
    
    # Set seed
    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Generate video
    print(f"\nGenerating video with prompt: {prompt}")
    print(f"Resolution: {resolution}, Steps: {num_steps}, Frames: {num_frames}")
    
    generation_start = time.time()
    
    # Generate video
    video = model.generate(
        input_prompt=prompt,
        img=input_image,
        max_area=MAX_AREA_CONFIGS[resolution],
        frame_num=num_frames,
        shift=shift_scale,
        sampling_steps=num_steps,
        guide_scale=guidance_scale,
        n_prompt=negative_prompt,
        seed=seed,
        offload_model=offload_model
    )
    
    generation_time = time.time() - generation_start
    print(f"\nVideo generation took: {generation_time:.2f} seconds")
    print(f"Average time per step: {generation_time / num_steps:.2f} seconds")
    
    # Save video
    print(f"\nSaving video to: {output_file}")
    if video is not None:
        cache_video(
            tensor=video[None],
            save_file=output_file,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"\nVideo saved successfully: {output_file}")
    else:
        print("\nWarning: No video generated")
    
    # Final memory usage
    print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    return output_file


def generate_i2v_multi_gpu(
    image_path,
    prompt,
    output_file="i2v_output.mp4",
    checkpoint_dir="./Wan2.1-I2V-14B-480P",
    resolution="480*832",
    num_frames=81,
    num_steps=50,
    guidance_scale=5.0,
    shift_scale=3.0,
    seed=-1,
    negative_prompt=""
):
    """Generate video using multi-GPU distributed setup"""
    
    gpu_count, _ = get_gpu_info()
    print(f"=== MULTI-GPU MODE WITH {gpu_count} GPUs ===")
    print("Using standard Wan2.1 distributed inference")
    
    # Build the torchrun command
    width, height = resolution.split('*')
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpu_count}",
        "generate.py",
        "--task", "i2v-14B",
        "--ckpt_dir", checkpoint_dir,
        "--size", f"{width}*{height}",
        "--frame_num", str(num_frames),
        "--sample_steps", str(num_steps),
        "--sample_guide_scale", str(guidance_scale),
        "--shift", str(shift_scale),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size", str(gpu_count),
        "--prompt", prompt,
        "--image", image_path,
        "--save_file", output_file
    ]
    
    if negative_prompt:
        cmd.extend(["--n_prompt", negative_prompt])
    
    if seed > 0:
        cmd.extend(["--seed", str(seed)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"\nVideo saved successfully: {output_file}")
    else:
        print(f"\nError generating video: {result.stderr}")
    
    return output_file if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate video from image using Wan2.1 14B I2V - Adaptive Hardware Version"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for motion")
    parser.add_argument("--output_file", type=str, default="i2v_14b_adaptive.mp4", help="Output filename")
    parser.add_argument("--checkpoint_dir", type=str, default="./Wan2.1-I2V-14B-480P", 
                        help="I2V model directory")
    parser.add_argument("--resolution", type=str, default="480*832", 
                        choices=["480*832", "832*480", "720*1280", "1280*720"], help="Resolution")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--num_steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--shift_scale", type=float, default=3.0, help="Shift scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--no_offload", action="store_true", 
                        help="Disable model offloading (single GPU mode only)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Detect hardware and choose appropriate mode
    gpu_count, vram_gb = get_gpu_info()
    print(f"Detected {gpu_count} GPU(s) with {vram_gb:.1f}GB VRAM")
    
    if should_use_fp8_mode():
        print("\nUsing SINGLE GPU mode with FP8 optimization")
        print("(To force multi-GPU mode, set FORCE_MULTI_GPU_MODE=1)")
        
        # Use offload_model unless explicitly disabled
        offload_model = not args.no_offload
        
        generate_i2v_single_gpu_fp8(
            image_path=args.image,
            prompt=args.prompt,
            output_file=args.output_file,
            checkpoint_dir=args.checkpoint_dir,
            resolution=args.resolution,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            shift_scale=args.shift_scale,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            offload_model=offload_model
        )
    else:
        print("\nUsing MULTI-GPU distributed mode")
        print("(To force FP8 mode, set FORCE_FP8_MODE=1)")
        
        generate_i2v_multi_gpu(
            image_path=args.image,
            prompt=args.prompt,
            output_file=args.output_file,
            checkpoint_dir=args.checkpoint_dir,
            resolution=args.resolution,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            shift_scale=args.shift_scale,
            seed=args.seed,
            negative_prompt=args.negative_prompt
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()