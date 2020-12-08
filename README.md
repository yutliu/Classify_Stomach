## Classification for Stomach
# Usage

### Prepare data

The data structure is as follows
```
├── RootData # train and validation data
	├── train
            ├── 0
                ├── 9205c1f6b52c11eabd01000c29e37e62.jpg
                ├──    ...
                ├── 9244250eb52c11eabd01000c29e37e62.jpg
            ├── 1
            ├──    ...
            ├── 28
    ├── val
        ├── 0
        ├── 1
        ├──    ...
        ├── 28
    ├── README.txt #index and class name
```

### Train

* If you want to train from scratch, you can run as follows:

```
python train.py --batch-size 256 --gpus 0,1,2,3
```

* If you want to train from one checkpoint, you can run as follows(for example train from `epoch_4.pth.tar`, the `--start-epoch` parameter is corresponding to the epoch of the checkpoint):

```
python train.py --batch-size 256 --gpus 0,1,2,3 --resume output/epoch_4.pth.tar --start-epoch 4
```