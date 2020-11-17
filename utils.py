import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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


#混淆矩阵
def plot_confusion_matrix(y_true, y_pred, save_path, epoch):
    confusion_mat = confusion_matrix(y_true,y_pred)
    class_number = max(y_true) + 1
    fig, ax = plt.subplots()
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(class_number)

    for label in ax.xaxis.get_ticklabels():
        # label is a Text instance
        label.set_rotation(45)
        label.set_fontsize(7)

    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.show()
    plt.savefig(os.path.join(save_path, f"epoch{epoch}.png"))


if __name__ == "__main__":
    y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    y_pred = [0, 0, 0, 28, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 25, 1, 0, 1, 1, 1, 1, 1, 1, 25, 25, 28, 24, 28]
    plot_confusion_matrix(y_true, y_pred, "output", 1)
    print(1)