from torchvision.datasets.voc import VOCDetection
import os
from data import AnnotationTransform
import cv2
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import numpy as np

class GeneralImageDataset(VOCDetection):

    def __init__(self, root_dir, image_set='trainval', transforms = None, class_names: list = None, keep_difficult=True):
        image_dir = os.path.join(root_dir, 'JPEGImages')
        annotation_dir = os.path.join(root_dir, 'Annotations')
        splits_dir = os.path.join(root_dir, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

        self.transforms = transforms
        self.class_names = ["__background__"] + class_names
        self.keep_difficult = keep_difficult

    def __getitem__(self, index):

        img = cv2.imread(self.images[index])
        target = ET.parse(self.annotations[index]).getroot()
        target = self._transform_target(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _transform_target(self, target):
        res = np.empty((0,5))
        for obj in target.iter('object'):

            diffucult_elem = obj.find('difficult')

            if diffucult_elem is None:
                difficult = False
            else:
                difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_names.index(name)
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]



if __name__ == '__main__':
    from configs.CC import Config
    from utils.core import config_compile
    from data import preproc
    cfg = Config.fromfile("./configs/m2det320_resnet101.py")
    config_compile(cfg)
    _preproc = preproc(cfg.model.input_size, cfg.model.rgb_means, cfg.model.p)
    dataset = GeneralImageDataset(root_dir="./dataset",
                                  class_names=cfg.model.m2det_config.class_names,
                                  transforms=_preproc)
    print(dataset[0])