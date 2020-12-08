import torch.nn as nn
import torch
from utils import plot_confusion_matrix, plot_surveychart
from collections import defaultdict
import os
import numpy as np
import cv2


def test_model(model, dataloader, logging, args, epoch):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    total_pre_number = 0
    all_pre_list = []
    all_labels_list = []
    # Iterate over data.
    for i, (inputs, labels) in enumerate(dataloader):

        with torch.no_grad():
        # wrap them in Variable
            inputs = inputs.cuda()
            labels = labels.cuda()
            # forward
            outputs = model(inputs)
            norm_outputs = nn.functional.softmax(outputs, dim=1)
            preds_score, preds = torch.max(norm_outputs.data, 1)
            over_index = preds_score > args.precision_conf

        over_index = over_index.cpu().numpy()
        total_pre_number += preds.shape[0]

        """Choose over precision_conf classify"""
        preds_list = preds.cpu().tolist()
        labels_list = labels.cpu().tolist()
        preds_list = [preds_list[i] for i, is_over in enumerate(over_index) if is_over]
        labels_list = [labels_list[i] for i, is_over in enumerate(over_index) if is_over]

        all_pre_list.extend(preds_list)
        all_labels_list.extend(labels_list)
        # statistics
        running_corrects += torch.sum(preds == labels.data).float()
    all_pre_list.extend(list(range(args.num_class)))
    all_labels_list.extend(list(range(args.num_class)))

    epoch_acc = running_corrects / total_pre_number
    logging.info('test Acc: {:.4f}'.format(epoch_acc))
    if args.save_vis_path != '':
        plot_confusion_matrix(all_labels_list, all_pre_list, args.save_vis_path, epoch, args.data_dir)
        plot_surveychart(all_labels_list, all_pre_list, args.save_vis_path, epoch, args.data_dir)
    return epoch_acc


def test_model_saveimg(model, dataloader, logging, args, epoch):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    total_pre_number = 0
    all_pre_list = []
    all_labels_list = []
    # Iterate over data.
    for i, (imgs_path, inputs, labels) in enumerate(dataloader):

        with torch.no_grad():
            # wrap them in Variable
            inputs = inputs.cuda()
            labels = labels.cuda()
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

        total_pre_number += preds.shape[0]
        all_pre_list.extend(preds.cpu().tolist())
        all_labels_list.extend(labels.cpu().tolist())
        # statistics
        running_corrects += torch.sum(preds == labels.data).float()

        label_txt = os.path.join(args.data_dir, "README.txt")
        if os.path.exists(label_txt):
            with open(label_txt, 'r', encoding="gbk", errors='ignore') as f:
                all_lines = f.readlines()
            class_names = [line.strip('\n').split(',') for line in all_lines]
            class_id2name = {class_name[0]:class_name[1] for class_name in class_names}
        else:
            assert 0, "{} does not exist".format(label_txt)

        for index, is_error in enumerate((preds == labels.data).cpu().numpy()):
            if not is_error:
                pred = preds[index].item()
                label = labels[index].item()
                prefix = imgs_path[index].split('/')[-1].split('.')[0]
                error_img_name = f"pre{class_id2name[f'{pred}']}_true{class_id2name[f'{label}']}_{prefix}.jpg"
                err_img = cv2.imread(imgs_path[index])
                cv2.imwrite(os.path.join(args.error_image_path, error_img_name), err_img)




    epoch_acc = running_corrects / total_pre_number

    logging.info('test Acc: {:.4f}'.format(epoch_acc))
    if args.save_vis_path != '':
        plot_confusion_matrix(all_labels_list, all_pre_list, args.save_vis_path, epoch, args.data_dir)
        plot_surveychart(all_labels_list, all_pre_list, args.save_vis_path, epoch, args.data_dir)
    return epoch_acc
