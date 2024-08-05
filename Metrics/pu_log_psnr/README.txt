This is a collection matlab code for simple quality metrics, which can be used with
HDR images.

There are two types of quality metrics:

a) Display-referred - those metrics expect that the values in images
correspond to the luminance emitted from the HDR display, on which
such images as displayed. They account for the fact that distortions
are less visible in darker image parts and should be generally more
accurate than luminance-independent metrics.

Display-referred metrics:

qm_pu2_psnr.m - PSNR computed in the perceptually uniform space
qm_pu2_ssim.m - SSIM computed in the perceptually uniform space
qm_pu2_msssim.m - MS-SSIM computed in the perceptually uniform space

More details on display-referred metrics can be found in:

Aydın, T. O., Mantiuk, R., & Seidel, H.-P. (2008).
Extending quality metrics to full luminance range images.
Proceedings of SPIE (p. 68060B–10). SPIE.
doi:10.1117/12.765095

When using those metrics in a research paper, please refer to the
paper above and the corresponding base-metric paper (SSIM and MS-SSIM).

b) Luminance-independent - those metrics accept any relative HDR pixel
values and give identical results when values are multiplied by a
constant.

Luminance-independent metrics:

qm_weber_rmse.m - Weber-like MSE
qm_log_psnr_y.m - PSNR computed for log-10 values of luminance
qm_log_psnr_rgb.m - PSNR computed for log-10 values of red, green and
                                   blue color channels

===============================================
Author: Rafal Mantiuk
License: MIT

Note that this package contains the code for SSIM and MS-SSIM from the
authors (Zhou Wang). The MIT licese does NOT apply to this part of the
code.

===============================================
Copyright (c) 2015, Rafal Mantiuk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.