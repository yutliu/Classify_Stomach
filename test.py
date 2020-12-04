import torch.nn as nn
import torch
from utils import plot_confusion_matrix, plot_surveychart


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
            _, preds = torch.max(outputs.data, 1)

        total_pre_number += preds.shape[0]
        all_pre_list.extend(preds.cpu().tolist())
        all_labels_list.extend(labels.cpu().tolist())
        # statistics
        running_corrects += torch.sum(preds == labels.data).float()
    epoch_acc = running_corrects / total_pre_number

    logging.info('test Acc: {:.4f}'.format(epoch_acc))
    if args.save_vis_path != '':
        plot_confusion_matrix(all_labels_list, all_pre_list, args.save_vis_path, epoch, args.data_dir)
        plot_surveychart(all_labels_list, all_pre_list, args.save_vis_path, epoch, args.data_dir)
    return epoch_acc


