import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
        'toll_20_train': {
            "data_dir": "TOLL20",
            "split": "train"
        },
        'toll_20_val': {
            "data_dir": "TOLL20",
            "split": "val"
        },
        'toll_20_trainval': {
            "data_dir": "TOLL20",
            "split": "trainval"
        },
        'toll_20_test': {
            "data_dir": "TOLL20",
            "split": "test"
        },
        'toll_50_train': {
            "data_dir": "TOLL50",
            "split": "train"
        },
        'toll_50_val': {
            "data_dir": "TOLL50",
            "split": "val"
        },
        'toll_50_trainval': {
            "data_dir": "TOLL50",
            "split": "trainval"
        },
        'toll_50_test': {
            "data_dir": "TOLL50",
            "split": "test"
        },
        'toll_100_train': {
            "data_dir": "TOLL100",
            "split": "train"
        },
        'toll_100_val': {
            "data_dir": "TOLL100",
            "split": "val"
        },
        'toll_100_trainval': {
            "data_dir": "TOLL100",
            "split": "trainval"
        },
        'toll_100_test': {
            "data_dir": "TOLL100",
            "split": "test"
        },
        'toll_200_train': {
            "data_dir": "TOLL200",
            "split": "train"
        },
        'toll_200_val': {
            "data_dir": "TOLL200",
            "split": "val"
        },
        'toll_200_trainval': {
            "data_dir": "TOLL200",
            "split": "trainval"
        },
        'toll_200_test': {
            "data_dir": "TOLL200",
            "split": "test"
        },
        'toll_500_train': {
            "data_dir": "TOLL500",
            "split": "train"
        },
        'toll_500_val': {
            "data_dir": "TOLL500",
            "split": "val"
        },
        'toll_500_trainval': {
            "data_dir": "TOLL500",
            "split": "trainval"
        },
        'toll_500_test': {
            "data_dir": "TOLL500",
            "split": "test"
        },
        'toll_80_t_train': {
            "data_dir": "TOLLtest",
            "split": "train"
        },
        'toll_80_t_val': {
            "data_dir": "TOLLtest",
            "split": "val"
        },
        'toll_80_t_trainval': {
            "data_dir": "TOLLtest",
            "split": "trainval"
        },
        'toll_80_t_test': {
            "data_dir": "TOLLtest",
            "split": "test"
        },
        'toll_20_t_train': {
            "data_dir": "TOLL20T",
            "split": "train"
        },
        'toll_20_t_val': {
            "data_dir": "TOLL20T",
            "split": "val"
        },
        'toll_20_t_trainval': {
            "data_dir": "TOLL20T",
            "split": "trainval"
        },
        'toll_20_t_test': {
            "data_dir": "TOLL20T",
            "split": "test"
        },
        'toll_50_t_train': {
            "data_dir": "TOLL50T",
            "split": "train"
        },
        'toll_50_t_val': {
            "data_dir": "TOLL50T",
            "split": "val"
        },
        'toll_50_t_trainval': {
            "data_dir": "TOLL50T",
            "split": "trainval"
        },
        'toll_50_t_test': {
            "data_dir": "TOLL50T",
            "split": "test"
        },
        'toll_100_t_train': {
            "data_dir": "TOLL100T",
            "split": "train"
        },
        'toll_100_t_val': {
            "data_dir": "TOLL100T",
            "split": "val"
        },
        'toll_100_t_trainval': {
            "data_dir": "TOLL100T",
            "split": "trainval"
        },
        'toll_100_t_test': {
            "data_dir": "TOLL100T",
            "split": "test"
        },
        'toll_200_t_train': {
            "data_dir": "TOLL200T",
            "split": "train"
        },
        'toll_200_t_val': {
            "data_dir": "TOLL200T",
            "split": "val"
        },
        'toll_200_t_trainval': {
            "data_dir": "TOLL200T",
            "split": "trainval"
        },
        'toll_200_t_test': {
            "data_dir": "TOLL200T",
            "split": "test"
        },
        'toll_500_t_train': {
            "data_dir": "TOLL500T",
            "split": "train"
        },
        'toll_500_t_val': {
            "data_dir": "TOLL500T",
            "split": "val"
        },
        'toll_500_t_trainval': {
            "data_dir": "TOLL500T",
            "split": "trainval"
        },
        'toll_500_t_test': {
            "data_dir": "TOLL500T",
            "split": "test"
        },
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)
        elif "toll" in name:
            toll_root = DatasetCatalog.DATA_DIR
            if 'TOLL_ROOT' in os.environ:
                toll_root = os.environ['TOLL_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(toll_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="TOLLDataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
