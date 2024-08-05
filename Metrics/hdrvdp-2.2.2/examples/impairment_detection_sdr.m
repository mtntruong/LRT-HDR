% This example demonstrates how HDR-VDP can be used to detect impairments
% in HDR images. 
%
% Note that the predicted visibility of introduced distortions may not
% match the visibility of those seen on the screen. To match the visibility, 
% you may need to adjust the parameters, such as the peak luminance of the
% display, viewing distance, screen resolution, and others. 

if ~exist( 'hdrvdp3', 'file' )
    addpath( fullfile( pwd, '..') );
end

% The input SDR images must have its peak value at 1.
% Note that this is a 16-bit image. Divide by 255 for 8-bit images.
I_ref = double(imread( 'wavy_facade.png' )) / (2^16-1);

% Find the angular resolution in pixels per visual degree:
% 30" 4K monitor seen from 0.5 meters
ppd = hdrvdp_pix_per_deg( 30, [3840 2160], 0.5 );

% Noise

% Create test image with added noise
noise = randn(size(I_ref,1),size(I_ref,2)) * 0.01;
I_test_noise = max( I_ref + repmat( noise, [1 1 3] ), 0.0001 );

% Note that the color encoding is set to 'sRGB-display' for SDR images
res_noise = hdrvdp( I_test_noise, I_ref, 'sRGB-display', ppd );

% Blur

% Create test image that is blurry
I_test_blur = imgaussfilt( I_ref, 0.7 );

res_blur = hdrvdp( I_test_blur, I_ref, 'sRGB-display', ppd, {} );


% Context image to show in the visualization. The context image should be
% in the linear space (gamma-decoded).
I_context = get_luminance( I_ref.^2.2 );

% Visualize images
% This size is not going to be correct because we are using subplot

clf
subplot( 2, 2, 1 );
imshow( I_test_noise );
title( 'Noisy image' );

subplot( 2, 2, 2 );
imshow( hdrvdp_visualize( res_noise.P_map, I_context ) );
title( 'Detection of noise' );

subplot( 2, 2, 3 );
imshow( I_test_blur );
title( 'Blurry image' );

subplot( 2, 2, 4 );
imshow( hdrvdp_visualize( res_blur.P_map, I_context ) );
title( 'Detection of blur' );

