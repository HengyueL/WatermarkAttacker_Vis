import argparse, os, torch
from imwatermark import WatermarkEncoder, WatermarkDecoder
from diffusers import ReSDPipeline


def main(args):
    device = torch.device("cuda")
    vis_root_dir = os.path.join(
        ".", "visualization_diffusion"
    )
    os.makedirs(vis_root_dir, exist_ok=True)
    # === Read in Orig Image ===
    img_orig_path = os.path.join(
        "examples", "ori_imgs", "000000000711.png"
    )

    # === Watermark encoder.decoder
    encoder = WatermarkEncoder()
    decoder = WatermarkDecoder('bits', 32)

    # === Diffusion Model ===
    diffusion_model = ReSDPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16"
    )
    diffusion_model.set_progress_bar_config(disable=True)
    diffusion_model.to(device)
    print("Diffusion Model Loaded.")

if __name__ == "__main__":
    print("A Script to play with the diffusion watermark attacker.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.'
    )
    args = parser.parse_args()
    main(args)
    print("All completed.")