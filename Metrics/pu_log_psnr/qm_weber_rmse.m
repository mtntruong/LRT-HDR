function Q = qm_weber_rmse( I1, I2 )
% Root-Mean-Square-Error of Weber-like ratios
%
% Q = qm_weber_mse( I1, I2 )
%
% I1, I2 - two HDR images. They could be either gray-scale of color.
%
% Copyright (c) 2014, Rafal Mantiuk <mantiuk@gmail.com>

I1 = max( I1, 1e-5 ); % To avoid division by 0
I2 = max( I2, 1e-5 );

Q1 = ( (I1-I2).^2 ) ./ (I1.^2 + I2.^2);
Q = mean(Q1(:));

end