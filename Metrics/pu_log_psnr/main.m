clc;clear
R1 = imread('re_8.jpg');
R2 = imread('GT_8.jpg');
Q1 = qm_pu2_psnr( double(R1), double(R2) )
Q2 = qm_pu2_ssim( double(R1), double(R2) )