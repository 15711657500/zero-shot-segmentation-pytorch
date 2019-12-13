from dataloaders.datasets import cityscapes, combine_dbs, pascal, pascal_toy, sbd, bdd100k, bdd_toy, nice, embedding #coco
from mypath import Path
from torch.utils.data import DataLoader

import fasttext
import os

def make_data_loader(args, **kwargs):
    '''
    classes = {
            'pascal' : ['aeroplane','bicycle','bird','boat',
                 'bottle','bus','car','cat',
                 'chair','cow','diningtable','dog',
                 'horse','motorbike','person','pottedplant',
                 'sheep','sofa','train','tvmonitor']
            }
    ft_dir = os.path.join(Path.db_root_dir('wiki'), 'wiki.en.bin')
    print("Loading fasttext embedding - ", end='')
    ft = fasttext.load_model(ft_dir)
    print("Done")
    '''

    if args.dataset == 'pascal' or args.dataset == 'pascal_toy':
        '''
        classes = classes['pascal']
        nft = {}
        for word in classes:
            nft[word] = ft[word]
        '''
        
        if args.dataset == 'pascal': p = pascal
        else: p = pascal_toy
        train_set = p.VOCSegmentation(args, split='train')
        val_set = p.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = val_loader

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
        '''
    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
        '''
    elif args.dataset == 'bdd' or args.dataset == 'bdd_toy':
        if args.dataset == 'bdd':
            train_set = bdd100k.BDD100kSegmentation(args, split='train')
            val_set = bdd100k.BDD100kSegmentation(args, split='val')
            test_set = bdd100k.BDD100kSegmentation(args, split='test')
        else:
            train_set = bdd_toy.BDD100kSegmentation(args, split='train')
            val_set = bdd_toy.BDD100kSegmentation(args, split='val')
            test_set = bdd_toy.BDD100kSegmentation(args, split='test')

        num_classes = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader, num_classes

    elif args.dataset == 'nice':
        dataset = nice.Nice(args)
        num_classes = dataset.NUM_CLASSES
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        return loader, num_classes

    elif args.dataset == 'embedding':        
        dataset = embedding.Embedding(args)
        num_classes = dataset.NUM_CLASSES
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        return loader, num_classes

    else:
        dataset = nice.Nice(args, root=args.dataset)
        num_classes = dataset.NUM_CLASSES
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        return loader, num_classes

