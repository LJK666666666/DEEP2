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
        'coco_2017_train': {
            "data_dir": "COCO2017/JPEGImages",
            "ann_file": "COCO2017/Annotations/instances_train2017.json"
        },
        'coco_2017_val': {
            "data_dir": "COCO2017/JPEGImages",
            "ann_file": "COCO2017/Annotations/instances_val2017.json"
        },
        # COCO 数据集，仅包含 VOC 的 20 个类别
        'coco_2017_train_voc': {
            "data_dir": "COCO2017/JPEGImages",
            "ann_file": "COCO2017/Annotations/instances_train2017_voc.json"
        },
        'coco_2017_val_voc': {
            "data_dir": "COCO2017/JPEGImages",
            "ann_file": "COCO2017/Annotations/instances_val2017_voc.json"
        },
    }

    @staticmethod
    def get(name):
        # 注意：先检查 coco，因为 coco_2017_val_voc 包含 "voc" 字符串
        if name.startswith("coco"):
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)
        elif name.startswith("voc"):
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
