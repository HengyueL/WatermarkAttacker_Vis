from hop_skip_jump import HopSkipJump
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import os
import cv2
import matplotlib.pyplot as plt
from imwatermark import WatermarkEncoder, WatermarkDecoder
from noise_layers.diff_jpeg import DiffJPEG
from tqdm import tqdm


def plot_image(image_arr, save_name):
    figure, ax = plt.subplots(ncols=1, nrows=1)
    ax.imshow(image_arr)
    plt.savefig(save_name)
    plt.close(figure)


# Rewrite The decider into tensor wise operation
class Detector(nn.Module):
    def __init__(self, gt, decoder, th=0.8) -> None:
        self.th = th
        self.decoder = decoder
        self.gt = torch.from_numpy(gt)

    def forward(self, input_tensor):
        # === Input tensor needs to be [0, 1] tensor
        input_tensor = torch.clamp(input_tensor, 0, 1)
        input_array = (input_tensor.detach().cpu().numpy() * 255).astype(np.uint8)[0, :, :, :]
        input_array = np.transpose(input_array, [1, 2, 0])
        decoded_message = self.decoder.decode(input_array, 'rivaGan')[np.newaxis, :]
        decoded_message = torch.from_numpy(decoded_message)
        bit_acc = 1 - torch.sum(torch.abs(decoded_message-self.gt), 1)/self.gt.shape[1]
        class_idx = torch.logical_or((bit_acc>self.th), (bit_acc<(1-self.th))).long()
        return F.one_hot(class_idx, num_classes=2)

    def predict(self, input_array):
        input_tensor = torch.from_numpy(input_array).to(dtype=torch.float)
        with torch.no_grad():
            return self.forward(input_tensor).cpu().numpy()
        

def WEvade_B_Q(args, watermarked_images, init_adv_images, detector, num_queries_ls, verbose=True):
    num_images = len(watermarked_images)
    norm = 'inf'
    attack = HopSkipJump(classifier=detector, targeted=False, norm=norm, max_iter=0, max_eval=1000, init_eval=5, batch_size=1)
    
    total_num_queries = 0
    saved_num_queries_ls = num_queries_ls.copy()
    es_ls = np.zeros((num_images))    # a list of 'es' in Algorithm 3
    es_flags = np.zeros((num_images)) # whether the attack has been early stopped
    num_natural_adv = 0
    num_early_stop = 0
    num_regular_stop = 0

    adv_images = init_adv_images.copy()
    best_adv = init_adv_images
    best_norms = np.ones((num_images))*1e8

    ### Algorithm
    max_iterations = 1000 # a random large number
    for i in range(int(max_iterations/args.iter_step)):
        adv_images, num_queries_ls = attack.generate(x=watermarked_images, x_adv_init=adv_images, num_queries_ls=num_queries_ls, resume=True) # use resume to continue previous attack
        if verbose:
            print("Step: {}; Number of queries: {}".format((i * args.iter_step), num_queries_ls))

        # save the best results
        avg_error = 0
        for k in range(len(adv_images)):
            if norm == 'inf':
                error = np.max(np.abs(adv_images[k] - watermarked_images[k]))
            else:
                error = np.linalg.norm(adv_images[k] - watermarked_images[k])

            if es_flags[k]==0: # update if the attack has not been early stopped
                if error<best_norms[k]:
                    best_norms[k] = error
                    best_adv[k] = adv_images[k]
                    es_ls[k] = 0
                else:
                    es_ls[k]+=1
            avg_error += best_norms[k]
        avg_error = avg_error/2 # [-1,1]->[0,1]
        if verbose:
            print("Adversarial images at step {}.".format(i * args.iter_step))
            print("Average best error in l_{} norm: {}\n".format(norm, avg_error/len(adv_images)))

        # stopping criteria
        # natural_adv
        for k in range(len(adv_images)):
            if best_norms[k]==0 and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += 0
                saved_num_queries_ls[k] = 0
                num_natural_adv += 1
        # regular_stop
        for k in range(len(adv_images)):
            if num_queries_ls[k]>=args.budget and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += num_queries_ls[k]
                saved_num_queries_ls[k] = num_queries_ls[k]
                num_regular_stop+=1
        # early_stop
        for k in range(len(adv_images)):
            if es_ls[k]==args.ES and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += num_queries_ls[k]
                saved_num_queries_ls[k] = num_queries_ls[k]
                num_early_stop += 1

        if np.sum(es_flags==0)==0:
            break
        attack.max_iter = args.iter_step

    assert np.sum(es_flags)==num_images
    assert num_natural_adv+num_regular_stop+num_early_stop==num_images
    assert np.sum(saved_num_queries_ls)==total_num_queries
    del attack

    if verbose:
        print("Number of queries used for each sample:")
        print(saved_num_queries_ls)

    return best_adv, saved_num_queries_ls


# Note: watermarked_images, labels and adv_images are in numpy arrays
def JPEG_initailization(watermarked_images, labels, detector, quality_ls, natural_adv=None, verbose=True):
    # JPEG initialization
    adv_images = watermarked_images.copy()
    num_images = len(watermarked_images)
    init_num_queries_ls = np.zeros((num_images)) # number of queries used for initialization
    flags = np.zeros((num_images))               # whether an adversarial example has been found
    if natural_adv is not None:
        flags[natural_adv] = 1

    for quality in tqdm(quality_ls): 
        jpeg_module = DiffJPEG(quality=quality).cuda()
        for k in range(len(adv_images)):
            if flags[k]==1: # pass
                continue
            init_num_queries_ls[k] += 1
            jpeg_image = torch.from_numpy(watermarked_images[k:k+1])
            jpeg_image_max = torch.max(jpeg_image)
            jpeg_image_min = torch.min(jpeg_image)
            jpeg_image = (jpeg_image-jpeg_image_min)/(jpeg_image_max-jpeg_image_min)
            jpeg_image = jpeg_module(jpeg_image.cuda()).detach().cpu()
            jpeg_image = jpeg_image*(jpeg_image_max-jpeg_image_min)+jpeg_image_min
            jpeg_image = jpeg_image.numpy()

            pred = detector.predict(jpeg_image)
            pred = np.argmax(pred,-1)[0]
            if pred!=labels[k]: # succeed
                adv_images[k] = jpeg_image
                flags[k]=1
        del jpeg_module
    print("Finish JPEG Initialization.")
    
    if verbose:
        print("Flags:", flags)

    return adv_images, init_num_queries_ls


def main(args):
    device = torch.device("cuda")
    vis_root_dir = os.path.join(
        ".", "visualizations-BQ"
    )
    os.makedirs(vis_root_dir, exist_ok=True)
    img_orig_path = os.path.join(
        "examples", "ori_imgs", "000000001442.png"
    )
    img_orig_bgr = cv2.imread(img_orig_path)
    # Visualize image
    vis_img = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)
    save_name = os.path.join(vis_root_dir, "image_orig.png")
    plot_image(vis_img, save_name)

    # === Generate GT watermrk ===
    watermark_str = "0" * 16 + "1" * 16
    watermark_gt = np.asarray(
        [int(i) for i in watermark_str]
    )
    watermark = watermark_str.encode('utf-8')
    watermark_label = watermark_gt[np.newaxis, :]

    # === Setup Encoder and Encode watermark ===
    encoder = WatermarkEncoder()
    decoder = WatermarkDecoder('bits', 32)

    encoder.set_watermark('bits', watermark)
    img_watermarked_bgr = img_orig_bgr
    for _ in range(1):
        img_watermarked_bgr = encoder.encode(img_watermarked_bgr, 'rivaGan')
        # img_watermarked_bgr = encoder.encode(img_watermarked_bgr, 'dwtDct')
    # Visualize watermarked image
    vis_img = cv2.cvtColor(img_watermarked_bgr, cv2.COLOR_BGR2RGB)
    save_name = os.path.join(vis_root_dir, "image_watermarked.png")
    plot_image(vis_img, save_name)

    # === Process the watermarked images to [0, 1] tensor
    img_watermarked_np = np.transpose(
        img_watermarked_bgr.astype(np.float32) / 255.,
        [2, 0, 1]
    )
    img_watermarked_np = img_watermarked_np[np.newaxis, :, :, :]
    img_watermarked_tensor = torch.from_numpy(img_watermarked_np)

    detector = Detector(watermark_label, decoder)
    quality_ls = [99,90,70,50,30,10,5,3,2,1]
    init_adv_images, num_queries_ls = JPEG_initailization(img_watermarked_np, np.asarray([1]), detector, quality_ls, natural_adv=None, verbose=True)

    # Vis InitAdv Image
    vis_img = init_adv_images[0, :, :, :]
    vis_img = np.stack(
        [vis_img[2, :, :], vis_img[1, :, :], vis_img[0, :, :]],
        axis=2
    )
    save_name = os.path.join(vis_root_dir, "img_init_adv.png")
    plot_image(vis_img, save_name)
    print()


if __name__ == "__main__":
    print("Opening sentence.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.'
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")