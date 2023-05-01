## Training
Extract `Training_Samples_ICIP.zip` to obtain the folder `Training_Samples_ICIP`, then run
```
python train.py --data_path=/path/to/Training_Samples
```
Please cancel the training process every 10 epochs then rerun to update the learning rate using the following commands
```
# After 10th epoch
python train.py --data_path=/path/to/Training_Samples_ICIP --resume=./checkpoints/epoch_10.pth --set_lr=1e-6
# After 20th epoch
python train.py --data_path=/path/to/Training_Samples_ICIP --resume=./checkpoints/epoch_20.pth --set_lr=1e-7
# After 30th epoch
python train.py --data_path=/path/to/Training_Samples_ICIP --resume=./checkpoints/epoch_30.pth --set_lr=1e-8
# Stop after 40th epoch and you are done
```
After the training process completes, you should use the weight named `epoch_40.pth` for testing. 
