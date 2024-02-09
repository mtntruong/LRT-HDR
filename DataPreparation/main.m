clear; clc; close all;
addpath(fullfile(pwd,'utils'));

% Data folder
root = './DATA';
hdr_list = dir(fullfile(root, 'HDR', '*.hdr'));
seq_path = fullfile(root, 'SEQ');

% Exposure bias
bias = [0 3 6]; % [0 3 6] for -3, 0, +3 EV; [0 2 4] for -2, 0, +2 EV
ev = 2 .^ bias; 
gamma = 2.2;

% Patch size and counter
width = 1:128:1665;
height = 1:128:769;
mat_idx = -1;

% Output folder
mkdir('./Train_Pairs/GT');
mkdir('./Train_Pairs/SEQ');
mkdir('./Train_Pairs/OMEGA');

% Parameters for SIFT-Flow
SIFTflowpara.alpha=2*255;
SIFTflowpara.d=40*255;
SIFTflowpara.gamma=0.005*255;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=2;
SIFTflowpara.topwsize=10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations= 30;
cellsize=3;
gridspacing=1;

for n = 1 : length(hdr_list)

    seq = hdr_list(n).name(1:6);

    % Read input sequence, img2 is the reference
    img1 = im2double(imread(fullfile(seq_path, [seq '_01.tif'])));
    img2 = im2double(imread(fullfile(seq_path, [seq '_02.tif'])));
    img3 = im2double(imread(fullfile(seq_path, [seq '_03.tif'])));

    % Balance EV for non-reference exposures
    img1_balanced = ((img1 .^ gamma ./ ev(1)) .* ev(2) ) .^ (1/gamma);
    img3_balanced = ((img3 .^ gamma ./ ev(3)) .* ev(2) ) .^ (1/gamma);

    % Calculate flow from EV-balanced exposures and warp the original
    sift1 = mexDenseSIFT(img2,cellsize,gridspacing);
    
    sift2 = mexDenseSIFT(img1_balanced,cellsize,gridspacing);
    [vx,vy,~]=SIFTflowc2f(sift1,sift2,SIFTflowpara);
    img1_warped = warpImage(img1,vx,vy);
    
    sift2 = mexDenseSIFT(img3_balanced,cellsize,gridspacing);
    [vx,vy,~]=SIFTflowc2f(sift1,sift2,SIFTflowpara);
    img3_warped = warpImage(img3,vx,vy);

    % Balance EV for warped non-reference exposures
    img1_warped_balanced = ((img1_warped .^ gamma ./ ev(1)) .* ev(2) ) .^ (1/gamma);
    img3_warped_balanced = ((img3_warped .^ gamma ./ ev(3)) .* ev(2) ) .^ (1/gamma);

    % Create Omega's (set of valid pixels)
    omega1 = create_omega(img1_warped, img1_warped_balanced, img2);
    omega2 = ones(size(img2));
    omega3 = create_omega(img3_warped, img3_warped_balanced, img2);

    % Generate patches
    seq_set = cat(4, img1_warped, img2, img3_warped);
    omega_set = cat(4, omega1, omega2, omega3);
    radiance_gt = hdrread(fullfile(hdr_list(n).folder, hdr_list(n).name));
    for w = 1 : length(width)
        for h = 1 : length(height)
            mat_idx = mat_idx + 1;
            mat_name = num2str(mat_idx, '%06d');

            % For ground-truth
            radiance = radiance_gt(height(h):height(h)+127,width(w):width(w)+127,:);
            R = radiance(:,:,1);
            R = repmat(R(:),1,3);
            G = radiance(:,:,2);
            G = repmat(G(:),1,3);
            B = radiance(:,:,3);
            B = repmat(B(:),1,3);
            radiance = luma(cat(3, R, G, B)*65535);
            save(['./Train_Pairs/GT/' mat_name '.mat'], 'radiance', '-v7.3');

            % LDR
            RS = zeros(16384,3); GS = zeros(16384,3); BS = zeros(16384,3);
            RO = zeros(16384,3); GO = zeros(16384,3); BO = zeros(16384,3);
            for j = 1:3
                omega = omega_set(:,:,:,j);
                omega = omega(height(h):height(h)+127,width(w):width(w)+127,:);

                % For Omega
                channel = omega(:,:,1);
                channel = channel(:);
                RO(:,j) = channel;

                channel = omega(:,:,2);
                channel = channel(:);
                GO(:,j) = channel;

                channel = omega(:,:,3);
                channel = channel(:);
                BO(:,j) = channel;

                % For SEQ
                ldr = seq_set(:,:,:,j);
                ldr = ldr(height(h):height(h)+127,width(w):width(w)+127,:);

                channel = ldr(:,:,1);
                rad_channel = channel .^ gamma / ev(j);
                RS(:,j) = rad_channel(:);

                channel = ldr(:,:,2);
                rad_channel = channel .^ gamma / ev(j);
                GS(:,j) = rad_channel(:);

                channel = ldr(:,:,3);
                rad_channel = channel .^ gamma / ev(j);
                BS(:,j) = rad_channel(:);

            end
            omega = cat(3, RO, GO, BO);
            omega(omega==255) = 1;
            radiance_seq = luma(cat(3, RS, GS, BS)*65535);
            save(['./Train_Pairs/SEQ/' mat_name '.mat'], 'radiance_seq', '-v7.3');
            save(['./Train_Pairs/OMEGA/' mat_name '.mat'], 'omega', '-v7.3');
        end
    end
end
