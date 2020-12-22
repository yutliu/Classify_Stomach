import torch.nn as nn
import torch
from utils import plot_confusion_matrix, plot_surveychart
from draw_ROC import draw_ROCcurve_3class, draw_ROCcurve_cancer, draw_ROCcurve_inflam
from collections import defaultdict
import os
import numpy as np
import cv2
import pandas as pd


def sort_img_and_result(labels, pres, img_paths, scores):
    """
    :param labels: list
    :param pres: list
    :param img_paths: list
    :param scores: np.array
    :return:
    """
    csvname = "csv_files/result_imgname.csv"
    csv_data = pd.read_csv(csvname, usecols=[0], header=None, encoding="utf-8")
    # cancer_pre = [1 if pre == 2 else 0 for pre in pres]
    imgs_name = csv_data.values
    imgs_name = np.squeeze(imgs_name)
    new_result = defaultdict(list)
    for index, img_path in enumerate(img_paths):
        img_name = img_path.split('/')[-1]
        # new_result[img_name] = [labels[index], *scores[index, :].tolist()]
        new_result[img_name] = [labels[index], pres[index]]
    with open("csv_files/model_result_pre.csv", "w") as f:
        for name in imgs_name:
            f.write("{},{},{}\n".format(name, *new_result[name]))
    assert 1


def test_model(model, dataloader, logging, args, epoch=0):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    total_pre_number = 0
    all_pre_list = []
    all_labels_list = []
    all_img_path = []
    roc_label = []
    roc_score = []
    # Iterate over data.
    for i, (img_paths, inputs, labels) in enumerate(dataloader):
        all_img_path.extend(img_paths)

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

        """ROC"""
        roc_batch_label = np.zeros((labels.cpu().numpy().shape[0], 3))
        for row_id, each_label in enumerate(labels.cpu().numpy()):
            roc_batch_label[row_id, each_label] = 1
        roc_label.extend(roc_batch_label)
        roc_score.extend(norm_outputs.cpu().numpy())

    # all_pre_list.extend(list(range(args.num_class)))
    # all_labels_list.extend(list(range(args.num_class)))
    roc_label = np.stack(roc_label, axis=0)
    roc_score = np.stack(roc_score, axis=0)

    epoch_acc = running_corrects / total_pre_number
    logging.info('test Acc: {:.4f}'.format(epoch_acc))
    if args.save_vis_path != '':
        # pass
        draw_ROCcurve_cancer(all_labels_list, roc_score, all_img_path)
        draw_ROCcurve_inflam(all_labels_list, roc_score, all_img_path)
        # sort_img_and_result(all_labels_list, all_pre_list, all_img_path, roc_score)
        # draw_ROCcurve_3class(roc_label, roc_score)
        # plot_confusion_matrix(all_labels_list, all_pre_list, args.save_vis_path, epoch, args.data_dir)
        # plot_surveychart(all_labels_list, all_pre_list, args.save_vis_path, epoch, args.data_dir)
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


def save_video(video_result_dict, save_path):
    """
    :param video_result_dict: key:origin_img_path(str), value:prediction
    :param save_path:
    :return:
    """
    sz = (1920, 1080)
    fps = 25
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    class_name = ["Normal esophagus", "Reflux esophagitis", "Early esophageal cancer"]
    for origin_path, pres in video_result_dict.items():
        video_name = origin_path.split("/")[-1]
        video_capture = cv2.VideoCapture(origin_path)

        assert video_capture.get(cv2.CAP_PROP_FRAME_COUNT) >= len(pres), \
            "The total number of frames is not equal to the number of predictions"

        videoWriter = cv2.VideoWriter(os.path.join(save_path, video_name), fourcc, fps, sz)
        for frame_pre in pres:
            success, frame = video_capture.read()
            if not success:
                break
            cv2.putText(frame, class_name[frame_pre], (700, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), thickness=2, )

            # cv2.namedWindow(video_name, 0)
            # cv2.resizeWindow(video_name, 640, 640)
            # cv2.imshow(video_name, frame)
            # cv2.waitKey(10)
            videoWriter.write(frame)
        cv2.destroyAllWindows()
        videoWriter.release()


#test video
def test_video(model, dataloader, logging, args):
    model.eval()
    video_result_dict = {}

    for i, (video_path, all_img) in enumerate(dataloader):
        all_pre_list = []
        frame_count = 1
        print("Dealing ", video_path[0])
        for inputs in all_img:
            # print("frame count: {}, total: {}".format(frame_count, len(all_img)))
            with torch.no_grad():
                # wrap them in Variable
                inputs = inputs.cuda()
                # forward
                outputs = model(inputs)
                norm_outputs = nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(norm_outputs.data, 1)

            all_pre_list.extend(preds.cpu().tolist())
            frame_count += 1
        video_result_dict[video_path[0]] = all_pre_list

    print("Saving all videos with prediction")
    save_video(video_result_dict, args.save_vis_path)
    print("All videos have been saved")
