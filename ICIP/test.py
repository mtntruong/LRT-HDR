import os
import cv2
import torch
import h5py
import copy
import argparse
import numpy as np
from model import RPCA_Net
from util import write_EXR, luma, inv_luma, matlab_style_gauss2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default='hdm',
        help='Choosing test data.',
    )
    return parser.parse_args()


def load_pretrained(path, N_iter):
    model = RPCA_Net(N_iter=N_iter)
    model = model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def omega_creator(inp_channel):
    Zth = 10
    channel = copy.deepcopy(inp_channel)
    channel = np.reshape(np.transpose(channel), (np.prod(channel.shape), 1))
    omega = ((channel >= Zth) & (channel <= 255 - Zth)) * 1
    return np.squeeze(omega)


def data_mat_creator(inp_channel, channel_idx, ev):
    channel = copy.deepcopy(inp_channel)
    channel = np.transpose(channel).flatten()
    rad_channel = (channel ** 2.2) / (2 ** ev)
    return rad_channel


def HDR_inference(model, inp_data, inp_omega):
    data = torch.from_numpy(copy.deepcopy(inp_data).astype('float32'))
    data = torch.unsqueeze(data, 0)
    data = data.cuda()
    omega = torch.from_numpy(copy.deepcopy(inp_omega).astype('float32'))
    omega = torch.unsqueeze(omega, 0)
    omega = omega.cuda()

    with torch.no_grad():
        R_prediction = model(data[:, :, :3], omega[:, :, :3])
        G_prediction = model(data[:, :, 3:6], omega[:, :, 3:6])
        B_prediction = model(data[:, :, 6:], omega[:, :, 6:])
        hdr_prediction = torch.cat([torch.unsqueeze(R_prediction, 1),
                                    torch.unsqueeze(G_prediction, 1),
                                    torch.unsqueeze(B_prediction, 1)], dim=1)

    hdr_patch = hdr_prediction.cpu().numpy()
    return hdr_patch


def create_images(img_folder, out_folder, ckpt_path, N_iter, img_width, img_height):
    # Load model
    model = load_pretrained(ckpt_path, N_iter)

    # Grid
    w_grid = [0]; h_grid = [0]

    while True:
        if h_grid[-1] + 128 < img_height:
            h_grid.append(h_grid[-1] + 128/2)
        if w_grid[-1] + 128 < img_width:
            w_grid.append(w_grid[-1] + 128/2)
        else:
            h_grid[-1] = img_height - 128
            w_grid[-1] = img_width - 128
            break
    w_grid = np.array(w_grid, dtype=np.uint16)
    h_grid = np.array(h_grid, dtype=np.uint16)

    # Gaussian mask 1D
    gm = np.round(matlab_style_gauss2D((1, 128), 16) * 40, 2)
    # Vertical split
    gm_ver = np.ones((1, 128))
    gm_ver[:, 0:64] = gm[:, 0:64]
    gm_ver = np.tile(gm_ver, (128, 1))
    gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
    gm_ver_inv = np.fliplr(copy.deepcopy(gm_ver))
    gm_ver_inv[:, 64:128, :] = 1 - gm_ver[:, 0:64, :]
    # Horizontal Split
    gm_hor = np.ones((128, 1))
    gm_hor[0:64, :] = np.transpose(gm[:, 0:64])
    gm_hor = np.tile(gm_hor, (1, 128))
    gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
    gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
    gm_hor_inv[64:128, :, :] = 1 - gm_hor[0:64, :, :]
    # Gaussian mask 2D
    gm_2d = np.round(matlab_style_gauss2D((128, 128), 20) * 2500, 2)
    gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
    gm_2d_inv = 1 - gm_2d

    # HDR reconstruction
    ev = [0.0, 3.0, 6.0]
    gt_img_name = sorted(os.listdir(os.path.join(img_folder, 'HDR')))
    for exr in gt_img_name:
        print(exr)
        seq = exr[0:6]

        # Storing final HDR image
        HDR = np.float32(np.zeros((img_height, img_width, 3)))
        
        # Read exposure times
        #with open(os.path.join(img_folder, 'EXP', seq + '.txt'), 'r') as f:
        #   ev = [float(line.strip()) for line in f if line]

        # Read image stack
        img_stack = np.zeros((img_height, img_width, 3, 3))
        for i in range(1, 4):
            img = cv2.imread(os.path.join(img_folder, 'SEQ', seq + '_' + str(i).zfill(2) + '.tif'), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.float32(img)
            img_stack[:, :, :, i-1] = img / 65535.0

        # Read omega stack
        omg_stack = np.zeros((img_height, img_width, 3, 3))
        for i in range(1, 4):
            omg = cv2.imread(os.path.join(img_folder, 'OMEGA', seq + '_' + str(i).zfill(2) + '.png'))
            omg = cv2.cvtColor(omg, cv2.COLOR_BGR2RGB)
            omg_stack[:, :, :, i - 1] = omg
        omg_stack[omg_stack > 0] = 1

        # Patch reconstruction
        i = 0; j = 0
        while i < len(h_grid):
            while j < len(w_grid):
                RS = np.zeros((16384, 3)); GS = np.zeros((16384, 3)); BS = np.zeros((16384, 3))
                RO = np.zeros((16384, 3)); GO = np.zeros((16384, 3)); BO = np.zeros((16384, 3))
                h = h_grid[i]
                w = w_grid[j]

                for k in range(3):

                    ldr = img_stack[h:h+128, w:w+128, :, k]
                    ldr = np.squeeze(ldr)

                    # For Omega
                    # RO[:, k] = omega_creator(ldr[:, :, 0])
                    # GO[:, k] = omega_creator(ldr[:, :, 1])
                    # BO[:, k] = omega_creator(ldr[:, :, 2])
                    omg = omg_stack[h:h+128, w:w+128, :, k]
                    omg = np.squeeze(omg)
                    RO[:, k] = np.reshape(np.transpose(omg[:, :, 0]), 16384)
                    GO[:, k] = np.reshape(np.transpose(omg[:, :, 1]), 16384)
                    BO[:, k] = np.reshape(np.transpose(omg[:, :, 2]), 16384)

                    # For SEQ
                    RS[:, k] = data_mat_creator(ldr[:, :, 0], 0, ev[k])
                    GS[:, k] = data_mat_creator(ldr[:, :, 1], 1, ev[k])
                    BS[:, k] = data_mat_creator(ldr[:, :, 2], 2, ev[k])

                omega = np.hstack((RO, GO, BO))
                data = luma(np.hstack((RS, GS, BS))*65535.0)

                # Patch inference and reshape
                hdr_columns = HDR_inference(model, data, omega)
                R = np.float32(np.transpose(np.reshape(np.transpose(hdr_columns[:, 0]).flatten(), (128, 128))))
                G = np.float32(np.transpose(np.reshape(np.transpose(hdr_columns[:, 1]).flatten(), (128, 128))))
                B = np.float32(np.transpose(np.reshape(np.transpose(hdr_columns[:, 2]).flatten(), (128, 128))))
                hdr_patch = np.stack((R, G, B), axis=2)

                # Stitching
                if i == 0 and j == 0:
                    HDR[h:h+128, w:w+128, :] = copy.deepcopy(hdr_patch)
                elif i == 0:
                    hdr_patch = np.multiply(hdr_patch, gm_ver)
                    HDR[h:h+128, w-64:w+64, :] = np.multiply(HDR[h:h+128, w-64:w+64, :], gm_ver_inv)
                    HDR[h:h+128, w+64:w+128, :] = copy.deepcopy(hdr_patch[:, 64:128, :])
                    HDR[h:h+128, w:w+64, :] = HDR[h:h+128, w:w+64, :] + hdr_patch[:, 0:64, :]
                elif j == 0:
                    hdr_patch = np.multiply(hdr_patch, gm_hor)
                    HDR[h-64:h+64, w:w+128, :] = np.multiply(HDR[h-64:h+64, w:w+128, :], gm_hor_inv)
                    HDR[h+64:h+128, w:w+128, :] = copy.deepcopy(hdr_patch[64:128, :, :])
                    HDR[h:h+64, w:w+128, :] = HDR[h:h+64, w:w+128, :] + hdr_patch[0:64, :, :]
                else:
                    if i == len(h_grid) - 1:
                        HDR[h+64:h+128, w+32:w+128, :] = copy.deepcopy(hdr_patch[64:128, 32:128, :])
                    elif j == len(w_grid) - 1:
                        HDR[h+32:h+128, w+64:w+128, :] = copy.deepcopy(hdr_patch[32:128, 64:128, :])
                    else:
                        HDR[h+64:h+128, w+64:w+128, :] = copy.deepcopy(hdr_patch[64:128, 64:128, :])
                    patch_2d = np.multiply(hdr_patch, gm_2d)
                    patch_2d_inv = np.multiply(HDR[h:h+128, w:w+128, :], gm_2d_inv)
                    HDR[h:h+128, w:w+128, :] = patch_2d + patch_2d_inv

                j = j + 1
            i = i + 1
            j = 0

        mask = np.squeeze(np.multiply(img_stack[:, :, :, 0], img_stack[:, :, :, 1]))
        mask[mask < 1] = 0
        HDR = inv_luma(HDR)/65535.0
        HDR[mask == 1] = 1.0
        write_EXR(os.path.join(out_folder, exr), HDR)


if __name__ == '__main__':
    opt_args = parse_args()

    if opt_args.data == 'hdm':
        if not os.path.exists('./HDM'):
            os.mkdir('./HDM')
        create_images('../Test_HDM',
                      './HDM',
                      './checkpoints/epoch_25.pth',
                      N_iter=10, img_width=1820, img_height=980)

    elif opt_args.data == 'hdrv':
        if not os.path.exists('./HDRv'):
            os.mkdir('./HDRv')
        create_images('../Test_HDRv',
                      './HDRv',
                      './checkpoints/epoch_25.pth',
                      N_iter=10, img_width=1280, img_height=720)

    else:
        print('Incorrect dataset.')
    print('done')
