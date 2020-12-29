import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
from pylab import mpl
mpl.rcParams['font.sans-serif'] = 'Arial'

def main():
    filename = "csv_files/trainval_acc.csv"
    save_dir = "savepng/"
    csv_data = pd.read_csv(filename)
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    for lines in csv_data.iterrows():
        lines = lines[1].values
        train_acc.append(lines[0])
        val_acc.append(lines[1])
        train_loss.append(lines[2])
        val_loss.append(lines[3])

    acc_max_epoch = np.argmax(val_acc)*2

    x = np.arange(0, len(train_acc)) * 2
    plt.plot(x, train_acc, color='blue', label='train')
    plt.plot(x, val_acc, color='green', label='validation')
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.ylim((min(val_acc) - 0.2, 1))
    plt.xlim(0, 20)
    plt.axvline(x=acc_max_epoch, color='k', linestyle='--', lw=1)
    plt.legend(loc="lower right")

    x_major_locator=MultipleLocator(2)
    #把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator=MultipleLocator(10)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.savefig(os.path.join(save_dir, "epoch_acc.png"), dpi=330)

    plt.figure()

    plt.plot(x, train_loss, color='blue', label='train')
    plt.plot(x, val_loss, color='green', label='validation')
    plt.xlabel("epoch")
    plt.ylabel('Loss')
    # plt.ylim((0, max(val_loss) + 4))
    plt.legend(loc="upper right")

    x_major_locator=MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 20)
    plt.axvline(x=acc_max_epoch, color='k', linestyle='--', lw=1)

    plt.savefig(os.path.join(save_dir, "epoch_loss.png"), dpi=330)
    plt.show()

    assert 1




if __name__ == "__main__":
    main()