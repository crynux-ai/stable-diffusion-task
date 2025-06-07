"""
RTX 5090 Graphics Card Simple Example
Demonstrates how to run Stable Diffusion inference tasks on RTX 5090 graphics card
"""

from sd_task import utils
from sd_task.task_runner import run_inference_task
from sd_task.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid


def main():
    print("=== RTX 5090 Stable Diffusion Example ===\n")
    
    # 1. Detect GPU Environment
    print("1. Detecting GPU Environment")
    accelerator = utils.get_accelerator()
    print(f"   Accelerator type: {accelerator}")
    if accelerator == "cuda":
        gpu_info = utils.get_gpu_info()
        if gpu_info:
            print(f"   GPU name: {gpu_info['gpu_name']}")
            print(f"   GPU memory: {gpu_info['gpu_memory_gb']}GB")
            
            # Detect if it's RTX 5090
            is_5090 = utils.is_rtx_5090()
            print(f"   RTX 5090: {'Yes' if is_5090 else 'No'}")
            
            if is_5090:
                # Enable RTX 5090 optimizations
                optimized = utils.optimize_for_rtx_5090()
                print(f"   RTX 5090 optimization: {'Enabled' if optimized else 'Failed'}")
    print()
    
    # 2. Basic Image Generation Example
    print("2. Running Basic Image Generation")
    
    args = {
        "version": "2.5.0",
        "base_model": {
            "name": "runwayml/stable-diffusion-v1-5"
        },
        "prompt": "a majestic dragon flying over a mystical forest, fantasy art, highly detailed, 8k",
        "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
        "task_config": {
            "num_images": 2,
            "safety_checker": False,
            "steps": 25,
            "cfg": 7.5,
            "seed": 12345,
            "image_width": 512,
            "image_height": 512
        }
    }
    print("   Generation parameters:")
    print(f"   - Model: {args['base_model']['name']}")
    print(f"   - Prompt: {args['prompt']}")
    print(f"   - Number of images: {args['task_config']['num_images']}")
    print(f"   - Inference steps: {args['task_config']['steps']}")
    print(f"   - Image size: {args['task_config']['image_width']}x{args['task_config']['image_height']}")
    print("   - Starting generation...")
    
    try:        # Run inference task
        images = run_inference_task(InferenceTaskArgs.model_validate(args))
        
        print(f"   ✓ Successfully generated {len(images)} images")
        
        # Save individual images
        for i, image in enumerate(images):
            image.save(f"./rtx_5090_example_{i+1}.png")
            print(f"   - Saved image: rtx_5090_example_{i+1}.png")
        
        # Create grid if multiple images
        if len(images) > 1:
            grid = make_image_grid(images, rows=1, cols=len(images))
            grid.save("./rtx_5090_example_grid.png")
            print(f"   - Saved grid image: rtx_5090_example_grid.png")
        
    except Exception as e:
        print(f"   ✗ Generation failed: {e}")
        return
    
    print()
      # 3. High-quality SDXL example (if RTX 5090)
    if utils.is_rtx_5090():
        print("3. Running High-Quality SDXL Generation (RTX 5090 Exclusive)")
        
        sdxl_args = {
            "version": "2.5.0",
            "base_model": {
                "name": "stabilityai/stable-diffusion-xl-base-1.0"
            },
            "prompt": "ultra realistic portrait of a cyberpunk warrior, neon cityscape background, cinematic lighting, photorealistic, 8k uhd",
            "negative_prompt": "cartoon, anime, low quality, blurry, distorted",
            "task_config": {
                "num_images": 1,
                "safety_checker": False,
                "steps": 30,
                "cfg": 8.0,
                "seed": 54321,
                "image_width": 1024,
                "image_height": 1024
            }
        }
        print("   SDXL generation parameters:")
        print(f"   - Model: {sdxl_args['base_model']['name']}")
        print(f"   - Image size: {sdxl_args['task_config']['image_width']}x{sdxl_args['task_config']['image_height']}")
        print(f"   - Inference steps: {sdxl_args['task_config']['steps']}")
        print("   - Starting generation...")
        
        try:
            sdxl_images = run_inference_task(InferenceTaskArgs.model_validate(sdxl_args))
            print(f"   ✓ Successfully generated SDXL image")
            
            sdxl_images[0].save("./rtx_5090_sdxl_example.png")
            print(f"   - Saved SDXL image: rtx_5090_sdxl_example.png")
            
        except Exception as e:
            print(f"   ✗ SDXL generation failed: {e}")
    print("\n=== Example Complete ===")
    print("Generated images have been saved to the current directory")
    
    if utils.is_rtx_5090():
        print("\nRTX 5090 Optimization Tips:")
        print("- TF32 acceleration enabled")
        print("- Flash Attention enabled")
        print("- Memory management optimized")
        print("- High-resolution fast generation supported")


if __name__ == "__main__":
    main()
