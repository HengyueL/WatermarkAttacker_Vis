import argparse
import torch, os
import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torchvision.transforms as tf
from model import get_net
# import warnings
# warnings.filterwarnings("ignore")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
dimension = 224


def plot_image(image_arr, save_name):
    figure, ax = plt.subplots(ncols=1, nrows=1)
    ax.imshow(image_arr)
    plt.savefig(save_name)
    plt.close(figure)


def bgr2rgb(img_bgr):
    """
        img_bgr.shape = (xx, xx, 3)
    """
    img_rgb = np.stack(
        [img_bgr[:, :, 2], img_bgr[:, :, 1], img_bgr[:, :, 0]], axis=2
    )
    return img_rgb


def uint8_to_float(img_orig):
    return img_orig.astype(np.float32) / 255.


def float_to_int(img_float):
    return (img_float * 255).round().astype(np.int16)


def img_np_to_tensor(img_np, device):
    img_np = np.transpose(img_np, [2, 0, 1])
    img_np = img_np[np.newaxis, :, :, :]
    img_tensor = torch.from_numpy(img_np).to(device, dtype=torch.float)
    return img_tensor


def main(args):
    device = torch.device("cuda")
    vis_root_dir = os.path.join(
        ".", "visualization_dip"
    )
    os.makedirs(vis_root_dir, exist_ok=True)

    # === Read in Orig Image ===
    img_orig_path = os.path.join(
        "examples", "ori_imgs", "000000001442.png"
    )
    img_orig_bgr = cv2.imread(img_orig_path)  # bgr image, unit8 code
    # img_orig_rgb = bgr2rgb(img_orig_bgr)
    # save_name = os.path.join(vis_root_dir, "img_orig.png")
    # plot_image(img_orig_rgb, save_name)

    # === Encode a watermark ===
    watermark_str = "0" * 16 + "1" * 16
    watermark_gt = np.asarray(
        [int(i) for i in watermark_str]
    )
    watermark = watermark_str.encode('utf-8')
    print("GT  watermark: ", watermark_str)
    # Set up encoder/decoder
    encoder = WatermarkEncoder()
    decoder = WatermarkDecoder('bits', 32)
    encoder.set_watermark('bits', watermark)
    # Encode watermark
    img_watermarked_bgr = img_orig_bgr  # packed rivaGan encoder works with bgr images
    for _ in range(1):
        img_watermarked_bgr = encoder.encode(img_watermarked_bgr, "rivaGan")
    # # Visualize watermarked image
    # vis_img = bgr2rgb(img_watermarked_bgr)
    # save_name = os.path.join(vis_root_dir, "image_watermarked.png")
    # plot_image(vis_img, save_name)
    # Decode watermark (for verification)
    psnr_orig_watermarked = compare_psnr(img_orig_bgr.astype(np.int16), img_watermarked_bgr.astype(np.int16), data_range=255)
    print("PSNR [Orig - Watermarked] - {%.04f}" % psnr_orig_watermarked)

    watermark_decoded = decoder.decode(img_watermarked_bgr, "rivaGan")
    watermark_decoded_str = "".join([str(i) for i in watermark_decoded.tolist()])
    print("Watermarked img Decoding: ", watermark_decoded_str)
    bitwise_acc = np.mean(watermark_decoded == watermark_gt)
    print("  Bitwise Acc. - {:.4f} \% ".format(bitwise_acc * 100))
    
    # === Construct DIP ===
    model = get_net().to(device, dtype=torch.float)
    img_watermarked_bgr_float = uint8_to_float(img_watermarked_bgr)
    img_watermarked_bgr_tensor = img_np_to_tensor(img_watermarked_bgr_float, device)  # DIP input tensor
    # DIP params
    show_every = 50
    exp_weight = 0.99  # Exponential smoothing factor
    total_iters = 3000
    LR = 0.01
    reg_noise_std = 1./30.
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=LR)
    loss_func = torch.nn.MSELoss()
    
    # Optimize
    model.train()
    out_avg = None
    for num_iter in range(total_iters):
        optimizer.zero_grad()
        out = model(img_watermarked_bgr_tensor)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
        total_loss = loss_func(out, img_watermarked_bgr_tensor)
        total_loss.backward()
        optimizer.step()

        # For monitor purpose
        img_recon = np.transpose(torch.clamp(out.detach().cpu(), 0, 1).numpy()[0, :, :, :], [1, 2, 0])
        img_recon_np_int = float_to_int(img_recon)
        img_recon_smoothed = np.transpose(torch.clamp(out_avg.detach().cpu(), 0, 1).numpy()[0, :, :, :], [1, 2, 0])
        img_recon_np_smoothed_int = float_to_int(img_recon_smoothed)

        # psnr_recon_watermarked = compare_psnr(img_watermarked_bgr.astype(np.int16), img_recon_np_int, data_range=255)
        # psnr_orig_recon = compare_psnr(img_orig_bgr.astype(np.int16), img_recon_np_int, data_range=255)
        # psnr_recon_watermarked_smoothed = compare_psnr(img_watermarked_bgr.astype(np.int16), img_recon_np_smoothed_int, data_range=255)
        # print("Iter [%d] - Loss [%.06f]" % (num_iter, total_loss.item()))
        # print("  <PSNR> --- rec|water %.02f --- rec|orig %.02f --- recon_sm|water %.02f " % (psnr_recon_watermarked, psnr_orig_recon, psnr_recon_watermarked_smoothed))

        # Checkpoint Vis
        if num_iter % show_every == 0 and num_iter > 0:
            psnr_recon_watermarked = compare_psnr(img_watermarked_bgr.astype(np.int16), img_recon_np_int, data_range=255)
            psnr_orig_recon = compare_psnr(img_orig_bgr.astype(np.int16), img_recon_np_int, data_range=255)
            psnr_recon_watermarked_smoothed = compare_psnr(img_watermarked_bgr.astype(np.int16), img_recon_np_smoothed_int, data_range=255)
            print("Iter [%d] - Loss [%.06f]" % (num_iter, total_loss.item()))
            print("  <PSNR> --- rec|water %.02f --- rec|orig %.02f --- recon_sm|water %.02f " % (psnr_recon_watermarked, psnr_orig_recon, psnr_recon_watermarked_smoothed))

            # Decode the recon and compute the bitwise-acc
            img_recon_bgr = img_recon_np_int.astype(np.uint8)
            recon_decoding = decoder.decode(img_recon_bgr)
            recon_decoding_str = "".join([str(i) for i in recon_decoding.tolist()])
            bitwise_acc = np.mean(recon_decoding == watermark_gt)
            print("  Recon Bitwise Acc. - {:.4f} \% ".format(bitwise_acc * 100))
            print("  Recon code: {}".format(recon_decoding_str))

            img_recon_rgb = bgr2rgb(img_recon_bgr)
            save_name = os.path.join(vis_root_dir, "img_recon_iter_%d.png" % num_iter)
            plot_image(img_recon_rgb, save_name)

            

if __name__ == "__main__":
    print("Opening sentence: test dip watermark purging.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.'
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")