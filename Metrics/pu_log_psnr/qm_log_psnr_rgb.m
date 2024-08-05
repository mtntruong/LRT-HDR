function Q = qm_log_psnr_rgb( I1, I2)
% PSNR of log10 HDR pixel values
%
% Q = qm_log_psnr( I1, I2 )
%
% I1, I2 - two HDR images, can be both colour or grayscale.
%
% The metrics computes error for each RGB channel. 
%
% Copyright (c) 2014, Rafal Mantiuk <mantiuk@gmail.com>

l_peak = log( 10000 ); % The peak is assumed to be 10,000 cd/m^2

lI1 = log( max(I1, 1e-5) );
lI2 = log( max(I2, 1e-5) );

MSE = sum((lI1(:)-lI2(:)).^2)/numel(lI1);

Q = 20 * log10( l_peak / sqrt(MSE) );

end
