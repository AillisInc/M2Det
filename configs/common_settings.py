dataset = dict(
    VOC = dict(
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets = [('2007', 'test')],
        ),
    COCO = dict(
        train_sets=[('2014', 'train')],
        eval_sets=[('2014', 'val')],
        test_sets=[('2014', 'val')],
        )
    )

import os
root_path = './dataset'
VOCroot = os.path.join(root_path,"VOCdevkit")
COCOroot = os.path.join(root_path,"coco")