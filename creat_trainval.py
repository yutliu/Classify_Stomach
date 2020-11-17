import cv2
import os
import shutil
from utils import getMirrorImage

dataroot = "/media/adminer/data/Medical/StomachClassificationDataset_small/"
new_dataroot = "/media/adminer/data/Medical/StomachClassificationDataset_smalltrainval/"


if __name__ == "__main__":
    if not os.path.exists(new_dataroot):
        os.mkdir(new_dataroot)
    else:
        shutil.rmtree(new_dataroot)
        os.mkdir(new_dataroot)
    newtrain_imgpath = os.path.join(new_dataroot, "train")
    newval_imgpath = os.path.join(new_dataroot, "val")
    os.mkdir(newtrain_imgpath)
    os.mkdir(newval_imgpath)

    allclass_dir = os.listdir(dataroot)
    allclass_dir = [class_dir for class_dir in allclass_dir if '.' not in class_dir]
    count = 1
    count_num = 1

    for class_dir in allclass_dir:
        class_path = os.path.join(dataroot, class_dir)
        if not os.path.exists(os.path.join(newtrain_imgpath, class_dir)):
            os.mkdir(os.path.join(newtrain_imgpath, class_dir))
        if not os.path.exists(os.path.join(newval_imgpath, class_dir)):
            os.mkdir(os.path.join(newval_imgpath, class_dir))
        allimg_name = os.listdir(class_path)
        if len(allimg_name) > 3000:
            allimg_name = allimg_name[:3000]
        trainimg_name = allimg_name[:int(len(allimg_name)*0.9)]
        valimg_name = allimg_name[int(len(allimg_name)*0.9):]
        assert len(allimg_name) == len(trainimg_name) + len(valimg_name), "train + val is error!"
        for img_name in trainimg_name:
            imgpath = os.path.join(class_path, img_name)
            assert os.path.exists(imgpath), f"{imgpath} is not exists!"
            img = cv2.imread(imgpath)
            crop_image = getMirrorImage(img)
            # print(count_num, " crop image shape:", crop_image.shape)
            # count_num += 1
            if crop_image.shape[0] != 0:
                cv2.imwrite(os.path.join(newtrain_imgpath, class_dir, img_name), crop_image)

        for img_name in valimg_name:
            imgpath = os.path.join(class_path, img_name)
            assert os.path.exists(imgpath), f"{imgpath} is not exists!"
            img = cv2.imread(imgpath)
            crop_image = getMirrorImage(img)
            # print(count_num, " crop image shape:", crop_image.shape)
            # count_num += 1
            if crop_image.shape[0] != 0:
                cv2.imwrite(os.path.join(newval_imgpath, class_dir, img_name), crop_image)

        print(f"class {class_dir} is end")



