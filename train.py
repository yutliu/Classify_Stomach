from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
from models.resnext import *
import argparse
from MedicalDataLoader import ImageNetData
from test import test_model
from logger import creat_logger
from models.resnet import resnet152
from efficientnet_pytorch import EfficientNet


def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes, logging):
    since = time.time()
    resumed = False

    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch+1,num_epochs):

        # Each epoch has a training and validation phase
        all_phase = dataloders.keys()
        all_phase = ['val']
        eval_value = 0.0
        for phase in all_phase:
            if phase == 'train':
                running_loss = 0.0
                running_corrects = 0
                model.train()  # Set model to training mode
                tic_batch = time.time()
                # Iterate over data.
                for i, (inputs, labels) in enumerate(dataloders[phase]):
                    # wrap them in Variable
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    else:
                        inputs, labels = inputs, labels

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize  if in training phase
                    loss.backward()
                    optimizer.step()
                    if args.start_epoch > 0 and (not resumed):
                        scheduler.step(args.start_epoch+1)
                        resumed = True
                    else:
                        scheduler.step(epoch)

                    # statistics
                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == labels.data).float()

                    batch_loss = running_loss / ((i+1)*args.batch_size)
                    batch_acc = running_corrects / ((i+1)*args.batch_size)

                    if i % args.print_freq == 0:
                        logging.info('[Epoch {}/{}]-[batch:{}/{}] lr:{:.6f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f} batch/sec'.format(
                              epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1, scheduler.get_lr()[0], phase,
                            batch_loss, batch_acc, args.print_freq/(time.time()-tic_batch)))
                        tic_batch = time.time()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            else:
                eval_value = test_model(model, dataloders[phase], logging, args, epoch)

        if (epoch+1) % args.save_epoch_freq == 0:
            torch.save(model, os.path.join(args.save_path, "epoch{}_eval{:.3f}.pth").format(epoch, eval_value))

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of ResNeXt")
    parser.add_argument('--data_dir', type=str, default="/media/adminer/data/Medical/StomachClassification_trainval_14classes/")
    # parser.add_argument('--data_dir', type=str, default="/media/adminer/data/Medical/imgenet-2/")
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--img_size', type=int, default=244)
    parser.add_argument('--num_class', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_epoch_freq', type=int, default=1)
    parser.add_argument('--save_path', type=str, default="output")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--confumatrix_path', type=str, default="output/", help="draw confusion matrix if not empty")
    parser.add_argument('--model', type=str, default="resnet152", help="choose model")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # read data
    dataloders, dataset_sizes = ImageNetData(args)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("Let's use gpu: {}".format(args.gpus))

    # get model
    if "resnext" in args.model:
        model = resnext101(num_classes=args.num_class, img_size=args.img_size)
    elif "resnet" in args.model:
        model = resnet152(pretrained=True, num_classes=args.num_class)
    elif "efficientnet" in args.model:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.num_class)
    else:
        assert 0, "model name does not exist!"

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume).module.state_dict()
            # base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            model.module.load_state_dict(checkpoint)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    optimizer_ft = optim.Adam(model.parameters(), lr=args.lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)
    logging = creat_logger()

    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    model = train_model(args=args,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=args.num_epochs,
                           dataset_sizes=dataset_sizes,
                           logging = logging)
