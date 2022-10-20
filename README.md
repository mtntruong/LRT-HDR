# LRT-HDR
Source code and data for the paper  
Deep Unrolled Low-Rank Tensor Completion for High Dynamic Range Imaging  
Truong Thanh Nhat Mai, Edmund Y. Lam, and Chul Lee  
IEEE Transactions on Image Processing, vol. 31, pp. 5774-5787, 2022  
https://doi.org/10.1109/TIP.2022.3201708

We also provide source code for the matrix completion version published in ICIP for references  
Ghost-Free HDR Imaging Via Unrolling Low-Rank Matrix Completion  
Truong Thanh Nhat Mai, Edmund Y. Lam, and Chul Lee  
2021 IEEE International Conference on Image Processing (ICIP)  
https://doi.org/10.1109/ICIP42928.2021.9506201

If you have any question, please open an issue.

# Source code
The proposed algorithm is implemented in Python with PyTorch framework.  
The instructions for testing routine will be updated soon (they are bad codes, now I'm trying to make them cleaner and easier to use). In the meantime, you may take a look at the implementation of the main network, or download the datasets and results.
## Preparation
### Download training/testing samples
[Download from Microsoft OneDrive](https://dguackr-my.sharepoint.com/:f:/g/personal/mtntruong_dgu_ac_kr/Eo87pbMBtLZHt03HZmJ0yIwB_VJ6X5ruXOKSNBgS-0tw-A)

The folder contains four ZIP files:
- Training_Samples.zip: 13000 training samples
- Training_Samples_ICIP.zip: 13000 training samples used in the ICIP paper
- HDM-HDR_Test_Samples.zip: Warped exposures of the HDM-HDR dataset
- HDRv_Test_Samples.zip: Warped exposures of the HDRv dataset

### Download pretrained weights
If you do not have time to retrain the network, you may use pretrained weights  
[Download from Microsoft OneDrive](https://dguackr-my.sharepoint.com/:f:/g/personal/mtntruong_dgu_ac_kr/EkFXsyWaoJVIttajp9CpxQ8Bg8j4iz7buSyObidTcZjtmw)

The folder contains two PTH files:
- LRT-HDR_net.pth: pretrained weight of LRT-HDR
- ICIP_net.pth: pretrained weight of the matrix completion network (ICIP paper)

### Required Python packages
Please use `env.yml` to create an environment in [Anaconda](https://www.anaconda.com/products/distribution)
```
conda env create -f env.yml
```
Then activate the environment
```
conda activate lrt
```
If you want to change the environment name, edit the first line of `env.yml` before creating the environment.

## Training
Extract `Training_Samples.zip` to obtain the folder `Training_Samples`, then training process can be started by executing
```
python train_auto.py --data_path=/path/to/Training_Samples
# or
python train_manual.py --data_path=/path/to/Training_Samples
```
While `train_auto.py` adjusts learning rate automatically, it usually yields worse performance (still better than competing algorithms). Using `train_manual.py` provides best results but you have to manually adjust learning rate. I have tried several ways to update learning rate during training, including `torch.optim.lr_scheduler`, but manually adjusting learning rate is always better.

When using `train_manual.py`, please cancel the training process every 10 epochs then rerun it to change the learning rate using the following commands
```
# After 10th epoch
python train_manual.py --data_path=/path/to/Training_Samples --resume=./checkpoints/epoch_10.pth --set_lr=1e-6
# After 20th epoch
python train_manual.py --data_path=/path/to/Training_Samples --resume=./checkpoints/epoch_20.pth --set_lr=1e-7
# After 30th epoch
python train_manual.py --data_path=/path/to/Training_Samples --resume=./checkpoints/epoch_30.pth --set_lr=1e-8
# Stop after 40th epoch and you are done
```
After the training process complete, you should use the weight named `epoch_40.pth` for testing.

## Testing
TODO

# HDR Dataset and results
## Download
[Download from Microsoft OneDrive](https://dguackr-my.sharepoint.com/:f:/g/personal/mtntruong_dgu_ac_kr/EmgWtrTX6nNMmNmWaZHX0EQBEcPAg2wvZJluOsneVNdOfg)

The folder contains two ZIP files:
- Datasets.zip: This file contains 187 and 32 multi-exposure image sets generated from the HDM-HDR and HDRv datasets, respectively, as described in the paper.
- All_Synthesized_Results.zip: We also provide HDR images synthesized by the proposed algorithm and all other competing algorithms, so that you can inspect the results to your heart's content without rerunning 10 algorithms.

## Difference between our dataset and that of NTIRE challenge on HDR imaging
As you may be aware, our dataset and that of [NTIRE challenge](https://doi.org/10.1109/CVPRW53098.2021.00078) are all generated from the videos of [HDM-HDR dataset](https://doi.org/10.1117/12.2040003). However, the data formats of the generated LDR and HDR images are different. Our dataset has the same format as the [Kalantari and Ramamoorthi's dataset](https://doi.org/10.1145/3072959.3073609), that means it is fully compatible with existing HDR algorithms that are designed for Kalantari and Ramamoorthi's dataset. We also generate an additional test set from [HDRv](https://doi.org/10.1016/j.image.2013.08.018) with the same format.

# Citation
If our research or dataset are useful for your research, please kindly cite our work
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
or if you prefer the low-rank matrix completion algorithm
```
@inproceedings{Mai2021,
    author={Mai, Truong Thanh Nhat and Lam, Edmund Y. and Lee, Chul},
    booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
    title={Ghost-Free HDR Imaging Via Unrolling Low-Rank Matrix Completion},
    year={2021},
    volume={},
    number={},
    pages={2928-2932},
    doi={10.1109/ICIP42928.2021.9506201}
}
```
Also, citing the original [HDM-HDR](https://doi.org/10.1117/12.2040003) and [HDRv](https://doi.org/10.1016/j.image.2013.08.018) video datasets is appreciated.
