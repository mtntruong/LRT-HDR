import os
import cv2
import torch
import h5py
import copy
import time
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
    parser.add_argument(
        '--data-path',
        default='./HDM-HDR_Test_Samples',
        help='Choosing test data.',
    )
    parser.add_argument(
        '--checkpoint',
        default='./LRT-HDR_net.pth',
        help='Choosing test data.',
    )
    parser.add_argument(
        '--output-path',
        default='./HDM-HDR_results',
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
    data = data.permute(0, 3, 2, 1).cuda()
    omega = torch.from_numpy(copy.deepcopy(inp_omega).astype('float32'))
    omega = torch.unsqueeze(omega, 0)
    omega = omega.permute(0, 3, 2, 1).cuda()

    start = time.time()
    with torch.no_grad():
        x_hat, hdr_prediction = model(data, omega)
    end = time.time() - start

    hdr_patch = hdr_prediction.cpu().numpy()
    return hdr_patch, end


def create_images(dataset, img_folder, out_folder, ckpt_path, N_iter):
    # Dataset's properties
    if dataset == 'hdm':
        img_height = 980
        img_width = 1820
        num_images = 55
    elif dataset == 'hdrv':
        img_height = 720
        img_width = 1280
        num_images = 32
    else:
        print('Incorrect dataset.')
        return 1
    
    # Load model
    model = load_pretrained(ckpt_path, N_iter)
    
    # Patch sizes
    psize = 128
    plength = psize * psize

    # Grid
    w_grid = [0]; h_grid = [0]

    while True:
        if h_grid[-1] + psize < img_height:
            h_grid.append(h_grid[-1] + psize)
        if w_grid[-1] + psize < img_width:
            w_grid.append(w_grid[-1] + psize)
        else:
            h_grid[-1] = img_height - psize
            w_grid[-1] = img_width - psize
            break
    w_grid = np.array(w_grid, dtype=np.uint16)
    h_grid = np.array(h_grid, dtype=np.uint16)

    # HDR reconstruction
    ev = [0.0, 3.0, 6.0] # exposure times
    for exr in range(num_images):
        
        seq = str(exr+1).zfill(6)
        print('Processing set ' + str(exr+1).zfill(6))

        # Storing final HDR image
        HDR = np.float32(np.zeros((img_height, img_width, 3)))

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
        total = 0
        i = 0; j = 0
        while i < len(h_grid):
            while j < len(w_grid):
                RS = np.zeros((plength, 3)); GS = np.zeros((plength, 3)); BS = np.zeros((plength, 3))
                RO = np.zeros((plength, 3)); GO = np.zeros((plength, 3)); BO = np.zeros((plength, 3))
                h = h_grid[i]
                w = w_grid[j]

                for k in range(3):

                    ldr = img_stack[h:h+psize, w:w+psize, :, k]
                    ldr = np.squeeze(ldr)

                    # For Omega
                    omg = omg_stack[h:h+psize, w:w+psize, :, k]
                    omg = np.squeeze(omg)
                    RO[:, k] = np.reshape(np.transpose(omg[:, :, 0]), plength)
                    GO[:, k] = np.reshape(np.transpose(omg[:, :, 1]), plength)
                    BO[:, k] = np.reshape(np.transpose(omg[:, :, 2]), plength)

                    # For SEQ
                    RS[:, k] = data_mat_creator(ldr[:, :, 0], 0, ev[k])
                    GS[:, k] = data_mat_creator(ldr[:, :, 1], 1, ev[k])
                    BS[:, k] = data_mat_creator(ldr[:, :, 2], 2, ev[k])

                omega = np.stack((RO, GO, BO), axis=2)
                data = luma(np.stack((RS, GS, BS), axis=2)*65535.0)

                # Patch inference and reshape
                hdr_columns, etime = HDR_inference(model, data, omega)
                total = total + etime
                R = np.float32(np.transpose(np.reshape(np.transpose(hdr_columns[0, :]).flatten(), (psize, psize))))
                G = np.float32(np.transpose(np.reshape(np.transpose(hdr_columns[1, :]).flatten(), (psize, psize))))
                B = np.float32(np.transpose(np.reshape(np.transpose(hdr_columns[2, :]).flatten(), (psize, psize))))
                hdr_patch = np.stack((R, G, B), axis=2)

                HDR[h:h+psize, w:w+psize, :] = copy.deepcopy(hdr_patch)

                j = j + 1
            i = i + 1
            j = 0

        print('Cost ' + str(total) + 's')
        
        # Applying OMEGA_2
        mask = np.squeeze(np.multiply(img_stack[:, :, :, 0], img_stack[:, :, :, 1]))
        mask[mask < 1] = 0
        HDR = inv_luma(HDR)/65535.0
        HDR[mask == 1] = 1.0
        img2 = cv2.imread(os.path.join(img_folder, 'SEQ', seq + '_02.tif'), -1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = np.float32(img2) / 65535.0
        radiance = img2 ** 2.2 / (2 ** ev[1])
        np.putmask(HDR, (img2 > 0.01) & (img2 < 0.99), radiance)
        
        # Final writing
        write_EXR(os.path.join(out_folder, str(exr+1).zfill(6) + '.exr'), HDR)


if __name__ == '__main__':
    opt_args = parse_args()

    os.makedirs(opt_args.output_path, exist_ok=True)
    create_images(opt_args.data,
                  opt_args.data_path,
                  opt_args.output_path,
                  opt_args.checkpoint,
                  N_iter=10)

    print('done')
