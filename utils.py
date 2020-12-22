import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import logging
import time

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Arial']



def creat_logger(phase):
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    save_path = "logs/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    filename = time.strftime("%Y%m%d_%H%M", time.localtime())
    fhlr = logging.FileHandler(os.path.join(save_path, f'{phase}_{filename}.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    logger.info('This is a Log')
    logging.getLogger('matplotlib.font_manager').disabled = True
    return logger


def getMirrorImage(orgin_img):
    gray_img = cv2.cvtColor(orgin_img, cv2.COLOR_BGR2GRAY)
    threshold = 5
    ret, thresh = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours == []:
        return np.array([])
    c1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(c1)
    if w*h < 100:
        return np.array([])
    # cv2.rectangle(orgin_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    crop_img = orgin_img[y:(y+h), x:(x + w)]

    #Visualization
    # cv2.drawContours(orgin_img, [c1], -1, (0, 0, 255), 3)
    # cv2.namedWindow("Image", flags=0)
    # cv2.resizeWindow("Image", 720, 480)
    # cv2.imshow('Image', crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return crop_img


def plot_confusion_matrix(y_true, y_pred, save_path, epoch, legend_path):
    if os.path.exists(os.path.join(legend_path, "README.txt")):
        with open(os.path.join(legend_path, "README.txt"), 'r', encoding="gbk", errors='ignore') as f:
            all_lines = f.readlines()
        class_names = [line.strip('\n').split(',')[1] for line in all_lines]
    else:
        assert 0, "{} does not exist".format(os.path.exists(os.path.join(legend_path, "README.txt")))

    #count confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)

    #average
    confusion_mat_norm = confusion_mat[:]
    confusion_mat_norm = confusion_mat_norm / np.sum(confusion_mat_norm, axis=1)

    fig, ax = plt.subplots()
    plt.rcParams['savefig.dpi'] = 330
    plt.rcParams['figure.dpi'] = 330
    im = ax.imshow(confusion_mat_norm, cmap=plt.cm.Blues)

    # show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    #  and label them with the respective list entries
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Prediction', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, confusion_mat[i, j],
                           ha="center", va="center", color="#1B1919FF")

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, f"epoch{epoch}_confumatrix.png"))
    # plt.show()
    assert 1


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
        example:
            results =
            {
                'class 1': [10, 15],
                'class 2': [26, 22],
            }
    category_names : list of str
        The category labels.
        example:
            category_names = ['Correct classification', 'Wrong classification']

    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('Blues')(
        np.linspace(0.85, 0.15, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='left', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    fig.suptitle("总体分类情况水平条形图")
    plt.tight_layout()
    return fig, ax


def plot_surveychart(y_true, y_pred, save_path, epoch, legend_path):
    """
    :param y_true: list include all classes label numbers
    :param y_pred: list include all classes pre numbers
    :param legend_path: class txt save path
    :param save_path: save survey chart path
    :return: None
    """
    if os.path.exists(os.path.join(legend_path, "README.txt")):
        with open(os.path.join(legend_path, "README.txt"), 'r', encoding="gbk", errors='ignore') as f:
            all_lines = f.readlines()
        class_names = [line.strip('\n').split(',') for line in all_lines]
        class_id2name = {class_name[0]:class_name[1] for class_name in class_names}
    else:
        assert 0, "{} does not exist".format(os.path.exists(os.path.join(legend_path, "README.txt")))

    survey_dict = defaultdict(list)
    max_classid = max(y_true) + 1
    every_img_pre = [y_true[i] == y_pred[i] for i in range(len(y_true))]
    for each_class in range(max_classid):
        temp = [y_true[i] for i, ea in enumerate(every_img_pre) if ea==True].count(each_class)
        survey_dict[class_id2name[f'{each_class}']].append(temp)
        survey_dict[class_id2name[f'{each_class}']].append(y_true.count(each_class) - temp)

    category_names = ['正确分类数量', '错误分类数量']
    survey(survey_dict, category_names)
    plt.savefig(os.path.join(save_path, f"epoch{epoch}_surveychart.png"))
    # plt.show()

def get_cropimg(origin_img):
    assert origin_img.shape[0] != 0, f"{origin_img.shape} is error"
    gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    threshold = 5
    ret, thresh = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        return np.array([])
    c1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(c1)
    if w * h < 100:
        return np.array([])
    crop_img = origin_img[y:(y + h), x:(x + w)]
    return crop_img


if __name__ == "__main__":
    y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]
    train_path = "/media/adminer/data/Medical/StomachClassification_trainval_14classes/"
    plot_confusion_matrix(y_true, y_pred, "output", 1, train_path)