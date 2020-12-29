from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
from models.resnext import *
import argparse
from MedicalDataLoader import medicalData
from test import test_model, test_model_saveimg, test_video
from utils import creat_logger
from models.resnet import resnet152, resnet101, resnet50
from efficientnet_pytorch import EfficientNet
from models.PMG.PMG_model import build_model as PMG
from models.resnet_nofc import resnet50_nofc


def main():
    parser = argparse.ArgumentParser(description="Classify Stomach")
    # parser.add_argument('--data_dir', type=str, default="/media/adminer/data/Medical/StomachClassification_trainval_14classes/")
    parser.add_argument('--data_dir', type=str, default="/media/adminer/data/Medical/StomachClassification_trainval_14classes/")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_class', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_epoch_freq', type=int, default=1)
    parser.add_argument('--save_path', type=str, default="output_14classes")
    parser.add_argument('--resume', type=str, default="savepths_14classes/epoch3_eval0.624.pth", help="For training from one checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--save_vis_path', type=str, default="vis", help="draw confusion matrix if not empty")
    parser.add_argument('--model', type=str, default="resnet", help="Choose model")
    parser.add_argument('--error_image_path', type=str, default="errorimg/", help="Save pictures with misclassification errors")
    parser.add_argument('--phase', type=str, default="val", help="trainval or val")
    parser.add_argument('--precision_conf', type=float, default=0.9, help="only choose > precision_conf result")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # if not os.path.exists(args.error_image_path):
    #     os.makedirs(args.error_image_path, exist_ok=True)

    if not os.path.exists(args.save_vis_path):
        os.makedirs(args.save_vis_path, exist_ok=True)

    # read data
    dataloders, dataset_sizes = medicalData(args)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("Let's use gpu: {}".format(args.gpus))

    # get model
    if "resnext" == args.model:
        model = resnext101(num_classes=args.num_class, img_size=args.img_size)
    elif "resnet" == args.model:
        model = resnet101(pretrained=True, num_classes=args.num_class)
    elif "efficientnet" == args.model:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.num_class)
    elif "PMG" == args.model:
        model = PMG(pretrained=True, num_classes=args.num_class)
    elif "resnet_nofc" == args.model:
        model = resnet50_nofc(num_classes=args.num_class)
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
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)
    logging = creat_logger(args.phase)

    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    model = train_model(args=args,
                        model=model,
                        criterion=criterion,
                        dataloders=dataloders,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.num_epochs,
                        dataset_sizes=dataset_sizes,
                        logging=logging)


def train_model(args, model, criterion, dataloders, optimizer, scheduler, num_epochs, dataset_sizes, logging):
    since = time.time()
    resumed = False

    best_model_wts = model.state_dict()

    if args.phase == "trainval":
        all_phase = dataloders.keys()
    elif args.phase == "val":
        all_phase = ['val']
    else:
        assert 0, "phase is error!"

    if args.phase == "val":
        eval_value = test_model(model, dataloders["val"], logging, args)
        # test_video(model, dataloders["val"], logging, args)
        return model


    for epoch in range(args.start_epoch+1,num_epochs):

        # Each epoch has a training and validation phase
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
                    if args.gpus is not None:
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
                        logging.info('[Epoch {}/{}]-[batch:{}/{}] lr:{:.2e} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f} batch/sec'.format(
                            epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1, scheduler.get_lr()[0], phase,
                            batch_loss, batch_acc, args.print_freq/(time.time()-tic_batch)))
                        tic_batch = time.time()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            else:
                eval_value = test_model(model, dataloders[phase], logging, args, epoch)
                # test_video(model, dataloders[phase], logging, args)


        if (epoch+1) % args.save_epoch_freq == 0:
            torch.save(model, os.path.join(args.save_path, "epoch{}_eval{:.3f}.pth").format(epoch, eval_value))

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    main()