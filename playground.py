import argparse
import torch, os
import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torchvision.transforms as tf


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
dimension = 224

def transform_img_tensor(image_tensor):
    transform = tf.Compose(
        [
            tf.Resize(dimension, antialias=True),
            tf.Normalize(mean=MEAN, std=STD)
        ]
    )
    return transform(image_tensor)


def plot_image(image_arr, save_name):
    figure, ax = plt.subplots(ncols=1, nrows=1)
    ax.imshow(image_arr)
    plt.savefig(save_name)
    plt.close(figure)


def convert_bgr_to_tensor(bgr_array, device):
    img_arr = np.stack(
            [bgr_array[:, :, 2], bgr_array[:, :, 1], bgr_array[:, :, 0]],
            axis=2
        )
    img_np = np.transpose(img_arr.astype(float) / 255, [2, 0, 1])[np.newaxis, :, :, :]
    img_tensor = torch.from_numpy(img_np)
    img_tensor = transform_img_tensor(img_tensor).to(device, dtype=torch.float)
    return img_tensor

def main(args):
    device = torch.device("cuda")
    vis_root_dir = os.path.join(
        ".", "visualizations"
    )
    os.makedirs(vis_root_dir, exist_ok=True)

    # === Read in Original Image ===
    img_orig_path = os.path.join(
        "examples", "ori_imgs", "000000000711.png"
    )
    # img_orig_path = os.path.join(
    #     "examples", "ori_imgs", "000000000776.png"
    # )
    img_orig_bgr = cv2.imread(img_orig_path)
    # Visualize image
    vis_img = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)
    save_name = os.path.join(vis_root_dir, "image_orig.png")
    plot_image(vis_img, save_name)

    # === Generate GT watermrk ===
    watermark_str = "0101010101010101" * 2
    watermark_gt = np.asarray(
        [int(i) for i in watermark_str]
    )
    watermark = watermark_str.encode('utf-8')

    # === Setup Encoder and Encode watermark ===
    encoder = WatermarkEncoder()
    encoder.set_watermark('bits', watermark) 
    img_watermarked_bgr = encoder.encode(img_orig_bgr, 'rivaGan')
    # Visualize watermarked image
    vis_img = cv2.cvtColor(img_watermarked_bgr, cv2.COLOR_BGR2RGB)
    save_name = os.path.join(vis_root_dir, "image_watermarked.png")
    plot_image(vis_img, save_name)

    # ===  Decode the watermarked image ===
    decoder = WatermarkDecoder('bits', 32)
    watermark_decoded = decoder.decode(img_watermarked_bgr, 'rivaGan')
    psrn_watermark = compare_psnr(img_orig_bgr, img_watermarked_bgr, data_range=255)
    print("Watermarked Image PSNR: ", psrn_watermark)

    # === Vis watermark Err image ===
    image_err = img_watermarked_bgr.astype(float) - img_orig_bgr.astype(float)
    print("Err Image Range: ", np.amin(image_err), np.amax(image_err))
    vis_img = np.stack(
        [image_err[:, :, 2], image_err[:, :, 1], image_err[:, :, 0]],
        axis=2
    )
    vis_img = (vis_img - np.amin(vis_img))/255.
    save_name = os.path.join(vis_root_dir, "image_err.png")
    plot_image(vis_img, save_name)

    # === Benchmarked by a noisy iamge
    magnitude = 10
    low, high = -magnitude, magnitude
    noisy_img_bgr = np.clip(np.random.randint(
        low, high, img_watermarked_bgr.shape
    ) + img_watermarked_bgr, 0, 255)
    # Visualize the noisy img
    vis_img = np.stack(
        [noisy_img_bgr[:, :, 2], noisy_img_bgr[:, :, 1], noisy_img_bgr[:, :, 0]],
        axis=2
    )
    save_name = os.path.join(vis_root_dir, "image_noisy.png")
    plot_image(vis_img, save_name)
    # Decode the noisy image
    watermark_noisy_decoded = decoder.decode(noisy_img_bgr, 'rivaGan')
    psnr_noisy_img = compare_psnr(img_watermarked_bgr, noisy_img_bgr, data_range=255)
    print("Noisy Image PSNR: ", psnr_noisy_img)

    # === Check Watermark Decoded ===
    print("Watermarks:")
    print("GT     : ", watermark_str)
    watermark_decoded_str = "".join([str(i) for i in watermark_decoded.tolist()])
    print("Decoded: ", watermark_decoded_str)
    watermark_noisy_str = "".join([str(i) for i in watermark_noisy_decoded.tolist()])
    print("Noisy  : ", watermark_noisy_str)


    # === Report Bitwise-Acc ===
    print("Calculate Bitwise-Acc: ")
    print("Decoded: {} %".format(np.mean(watermark_decoded == watermark_gt)*100))
    print("Noisy  : {} %".format(np.mean(watermark_noisy_decoded == watermark_gt) * 100))


    # === Get DINO model and check the DINO encoding ===
    dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    # Orig Image
    img_orig_input = convert_bgr_to_tensor(img_orig_bgr, device)
    img_orig_dino_feature = dino_backbone(img_orig_input)

    # Watermarked image
    img_watermarked_input = convert_bgr_to_tensor(img_watermarked_bgr, device)
    img_watermarked_dino_feature = dino_backbone(img_watermarked_input)

    distance_watermarked = torch.linalg.norm((img_watermarked_dino_feature - img_orig_dino_feature), ord=2)
    print("Dino Feature Distance")
    print("  Watermarked: ", distance_watermarked.item())

    # Noisy image
    img_noisy_input = convert_bgr_to_tensor(noisy_img_bgr, device)
    img_noisy_dino_feature = dino_backbone(img_noisy_input)
    distance_noisy = torch.linalg.norm((img_noisy_dino_feature - img_orig_dino_feature), ord=2)
    print("  NoisyImg:  ", distance_noisy.item())


if __name__ == "__main__":
    print("Opening sentence.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.'
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")