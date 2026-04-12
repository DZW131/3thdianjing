from torch.utils.data import DataLoader

from dataloaders.datasets import (
    beiertongbxr_region,
    cityscapes,
    coco,
    combine_dbs,
    feiai_region,
    her2_region,
    jijie,
    pascal,
    prostate_tls,
    qidai_region,
    sbd,
    taimo_region,
)


def _build_loader(dataset, batch_size, shuffle, **kwargs):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def make_data_loader(args, **kwargs):
    eval_batch_size = args.test_batch_size or args.batch_size

    if args.dataset == "pascal":
        train_set = pascal.VOCSegmentation(args, split="train")
        val_set = pascal.VOCSegmentation(args, split="val")
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=["train", "val"])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, None, num_class

    if args.dataset == "cityscapes":
        train_set = cityscapes.CityscapesSegmentation(args, split="train")
        val_set = cityscapes.CityscapesSegmentation(args, split="val")
        test_set = cityscapes.CityscapesSegmentation(args, split="test")
        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        test_loader = _build_loader(test_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, test_loader, num_class

    if args.dataset == "coco":
        train_set = coco.COCOSegmentation(args, split="train")
        val_set = coco.COCOSegmentation(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, None, num_class

    if args.dataset == "her2_region":
        train_set = her2_region.Her2Segmentation(args, split="train")
        val_set = her2_region.Her2Segmentation(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, None, num_class

    if args.dataset == "feiai_region":
        train_set = feiai_region.FeiaiSegmentation(args, split="train")
        val_set = feiai_region.FeiaiSegmentation(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, None, num_class

    if args.dataset == "beiertongbxr_region":
        train_set = beiertongbxr_region.BeiertongbxrSegmentation(args, split="train")
        val_set = beiertongbxr_region.BeiertongbxrSegmentation(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, None, num_class

    if args.dataset == "qidai_region":
        train_set = qidai_region.QidaiSegmentation(args, split="train")
        val_set = qidai_region.QidaiSegmentation(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, None, num_class

    if args.dataset == "taimo_region":
        train_set = taimo_region.TaimoSegmentation(args, split="train")
        val_set = taimo_region.TaimoSegmentation(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, None, num_class

    if args.dataset == "prostate_tls":
        train_set = prostate_tls.FeiaiSegmentation(args, split="train")
        val_set = prostate_tls.FeiaiSegmentation(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, None, num_class

    if args.dataset == "jijie":
        train_set = jijie.JijieSegmentation(args, split="train")
        val_set = jijie.JijieSegmentation(args, split="val")
        test_loader = None
        try:
            test_set = jijie.JijieSegmentation(args, split="test")
            test_loader = _build_loader(test_set, eval_batch_size, False, **kwargs)
        except FileNotFoundError:
            print("[Data] jijie test split not found; evaluation will fall back to the validation split.")

        num_class = train_set.NUM_CLASSES
        train_loader = _build_loader(train_set, args.batch_size, True, **kwargs)
        val_loader = _build_loader(val_set, eval_batch_size, False, **kwargs)
        return train_loader, val_loader, test_loader, num_class

    raise NotImplementedError("Unknown dataset: {}".format(args.dataset))


__all__ = ["make_data_loader"]
