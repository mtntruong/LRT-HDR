# LRT-HDR
Source code and data for our paper:  
Deep Unrolled Low-Rank Tensor Completion for High Dynamic Range Imaging  
Truong Thanh Nhat Mai, Edmund Y. Lam, and Chul Lee  
IEEE Transactions on Image Processing, 2022  
https://dx.doi.org/10.1109/TIP.2022.3201708

If you have any question, please open an issue.

# Source code
The proposed algorithm is implemented in Python with PyTorch framework.  
The full source code with training and testing routines will be updated soon. In the meantime, you may take a look at the implementation of the main network, or download the datasets and results.

# Dataset
## Download
Please visit the link below

[Download from Microsoft OneDrive](https://dguackr-my.sharepoint.com/:f:/g/personal/mtntruong_dgu_ac_kr/EmgWtrTX6nNMmNmWaZHX0EQBEcPAg2wvZJluOsneVNdOfg)

The folder contains two ZIP files
- Datasets.zip: This file contains 187 and 32 multi-exposure image sets generated from the HDM-HDR and HDRv datasets, respectively, as described in the paper.
- All_Synthesized_Results.zip: We also provide HDR images synthesized by the proposed algorithm and all other competing algorithms, so that you can inspect the results to your heart's content without rerunning 10 algorithms.

## Difference between our dataset and that of NTIRE challenges on HDR imaging [1, 2]
As you may be aware, our dataset and that of [1, 2] are all generated from the videos of HDM-HDR dataset [3]. However, the data formats of the generated LDR and HDR images are different. Our dataset has the same format as the Kalantari and Ramamoorthi's dataset [4], that means it is fully compatible with existing HDR algorithms that are designed for Kalantari and Ramamoorthi's dataset [4]. We also generate an additional test set from HDRv [5] with the same format.

**References**

[1] E. Pérez-Pellitero, S. Catley-Chandar, R. Shaw, A. Leonardis, R. Timofte *et al.*, “NTIRE 2022 challenge on high dynamic range imaging: Methods and results,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops*, Jun. 2022, pp. 1009–1023.

[2] E. Pérez-Pellitero, S. Catley-Chandar, A. Leonardis, R. Timofte *et al.*, “NTIRE 2021 challenge on high dynamic range imaging: Dataset, methods and results,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops*, Jun. 2021, pp. 691-700.

[3] J. Froehlich, S. Grandinetti, B. Eberhardt, S. Walter, A. Schilling, and H. Brendel, “Creating cinematic wide gamut HDR-video for the evaluation of tone mapping operators and HDR-displays,” in *Proc. SPIE*, vol. 9023, Mar. 2014, pp. 279–288.

[4] N. K. Kalantari and R. Ramamoorthi, “Deep high dynamic range imaging of dynamic scenes,” *ACM Trans. Graph.*, vol. 36, no. 4, pp.144:1–144:12, Jul. 2017.

[5] J. Kronander, S. Gustavson, G. Bonnet, A. Ynnerman, and J. Unger, “A unified framework for multi-sensor HDR video reconstruction,” *Signal Process. Image Commun.*, vol. 29, no. 2, pp. 203–215, Feb. 2014.

## Citation
If this dataset is useful for your research, please cite our work

```
@article{Mai2022,
    author={Mai, Truong Thanh Nhat and Lam, Edmund Y. and Lee, Chul},
    journal={IEEE Transactions on Image Processing}, 
    title={Deep Unrolled Low-Rank Tensor Completion for High Dynamic Range Imaging}, 
    year={2022},
    volume={31},
    number={},
    pages={5774-5787},
    doi={10.1109/TIP.2022.3201708}
}
```
and the respective authors of the original HDR video datasets
```
% HDM-HDR
@inproceedings{Froehlich2014,
    author = {J. Froehlich and S. Grandinetti and B. Eberhardt and S. Walter and A. Schilling and H. Brendel},
    title = {{Creating cinematic wide gamut HDR-video for the evaluation of tone mapping operators and HDR-displays}},
    volume = {9023},
    booktitle = {Proceedings of SPIE},
    pages = {279-288},
    year = {2014},
    month = {Mar.},
}
% HDRv
@article{Kronander2014,
    author = {J. Kronander and S. Gustavson and G. Bonnet and A. Ynnerman and J. Unger},
    title = {A unified framework for multi-sensor {HDR} video reconstruction},
    journal = {Signal Processing: Image Communication},
    volume = {29},
    number = {2},
    pages = {203-215},
    year = {2014},
    month = {Feb.},
}
```
