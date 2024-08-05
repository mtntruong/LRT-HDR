function Q = qm_log_psnr_y( I1, I2 )
% PSNR of log10 HDR pixel values
%
% Q = qm_log_psnr_y( I1, I2 )
%
% I1, I2 - two HDR images, can be both colour or grayscale.
%
% This is a luminance independent metric.
%
% Copyright (c) 2014, Rafal Mantiuk <mantiuk@gmail.com>

Q = qm_log_psnr_rgb( get_luminance(I1), get_luminance(I2) );

end

function Y = get_luminance( img )
% Return 2D matrix of luminance values for 3D matrix with an RGB image

dims = find(size(img)>1,1,'last');

if( dims == 3 )
    Y = img(:,:,1) * 0.212656 + img(:,:,2) * 0.715158 + img(:,:,3) * 0.072186;
elseif( dims == 1 || dims == 2 )
    Y = img;
else
    error( 'get_luminance: wrong matrix dimension' );
end

end
