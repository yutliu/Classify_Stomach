from torchvision import transforms
import os
import torch
from PIL import Image
import cv2
from utils import get_cropimg

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def medicalData(args):
    # data_transform, pay attention that the input of Normalize() is
    # Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize((1080, 1080)),
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.CenterCrop((1920, 1080)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}

    # image_datasets['train'] = MedicalTrainDataSet(os.path.join(args.data_dir, "train"), data_transforms['train'])
    image_datasets['train'] = MedicalTrainDataSet(os.path.join(args.data_dir, "train"), data_transforms['train'])

    image_datasets['val'] = MedicalValDataSet(os.path.join(args.data_dir, "val"), data_transforms['val'])
    # image_datasets['val'] = MedicalVideoDataSet(os.path.join(args.data_dir, "video"), data_transforms['val'])

    """train and val data and label"""
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True if x == 'train' else False,
                                                 drop_last=False,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes

class MedicalTrainDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_transforms):
        dirpath = os.listdir(root_dir)
        self.data_transforms = data_transforms
        self.root_dir = root_dir
        self.classdir = []
        for eachdir in dirpath:
            if os.path.isdir(os.path.join(root_dir, eachdir)):
                self.classdir.append(eachdir)
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        data, label = self.imgs[item]
        img = Image.open(data)
        # img = cv2.imread(data)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label

    def _make_dataset(self):
        images = []
        classname = list(map(int, self.classdir))
        classname.sort()
        for eachclass in classname:
            # if eachclass not in [0, 1]:
            #     break
            classdir_path = os.path.join(self.root_dir, str(eachclass))
            for num, eachimg in enumerate(os.listdir(classdir_path)):
                if num > 3000:
                    break
                imgpath = os.path.join(classdir_path, eachimg)
                assert os.path.exists(imgpath), f"{imgpath} is not exists!"
                assert str(eachclass) in imgpath, f"class is match img!"
                if self._is_image_file(imgpath):
                    item = (imgpath, eachclass)
                    images.append(item)
        return images

    def _is_image_file(self, filename):
        """Checks if a file is an image.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)



class MedicalValDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_transforms):
        dirpath = os.listdir(root_dir)
        self.data_transforms = data_transforms
        self.root_dir = root_dir
        self.classdir = []
        for eachdir in dirpath:
            if os.path.isdir(os.path.join(root_dir, eachdir)):
                self.classdir.append(eachdir)
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        data, label = self.imgs[item]
        img = Image.open(data)
        # img = cv2.imread(data)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return data, img, label

    def _make_dataset(self):
        # class_to_idx = self.label_dic
        images = []
        classname = list(map(int, self.classdir))
        classname.sort()
        for eachclass in classname:
            # if eachclass not in [0, 1]:
            #     break
            classdir_path = os.path.join(self.root_dir, str(eachclass))
            for index, eachimg in enumerate(os.listdir(classdir_path)):
                imgpath = os.path.join(classdir_path, eachimg)
                assert os.path.exists(imgpath), f"{imgpath} is not exists!"
                assert str(eachclass) in imgpath, f"class is match img!"
                if self._is_image_file(imgpath):
                    item = (imgpath, eachclass)
                    images.append(item)
        return images

    def _is_image_file(self, filename):
        """Checks if a file is an image.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


class MedicalVideoDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_transforms):
        self.all_video_name = os.listdir(root_dir)
        self.data_transforms = data_transforms
        self.root_dir = root_dir

        #save time
        # self.all_video_name = [self.all_video_name[1]]

    def __len__(self):
        return len(self.all_video_name)

    def __getitem__(self, item):
        video_name = self.all_video_name[item]
        video_path = os.path.join(self.root_dir, video_name)

        all_img = []
        video_capture = cv2.VideoCapture(video_path)
        while True:
            success, img = video_capture.read()
            if not success:
                break
            img = get_cropimg(img)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.data_transforms is not None:
                try:
                    img = self.data_transforms(img)
                except:
                    print("Cannot transform image")
            all_img.append(img)
        return video_path, all_img

