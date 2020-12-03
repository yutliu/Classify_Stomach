import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


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

    #count confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_mat, cmap=plt.cm.Blues)

    # show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    #  and label them with the respective list entries
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, confusion_mat[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("胃镜部位分类混淆矩阵")
    fig.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, f"epoch{epoch}.png"))


if __name__ == "__main__":
    y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]
    train_path = "/media/adminer/data/Medical/StomachClassification_trainval_14classes/"
    plot_confusion_matrix(y_true, y_pred, "output", 1, train_path)