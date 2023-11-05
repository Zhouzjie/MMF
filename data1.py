
class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, transform=None):
        self.vocab = vocab
        self.transform = transform
        loc = data_path + '/'
        self.newloc = 'D:/pytorch/MMF/data/' + 'flickr30k_images' + '/'
        self.dataset = jsonmod.load(open('D:/pytorch/MMF/dsran_data/data/f30k/dataset_flickr30k.json', 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == data_split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]
        # load the raw captions
        self.captions = []


        # -------- The main difference between python2.7 and python3.6 --------#
        # The suggestion from Hongguang Zhu (https://github.com/KevinLight831)
        # ---------------------------------------------------------------------#
        # for line in open(loc+'%s_caps.txt' % data_split, 'r', encoding='utf-8'):
        #     self.captions.append(line.strip())

        for line in open(loc+'%s_caps.txt' % data_split, 'rb'):
            self.captions.append(line.strip())

        # load the image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'val':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div # == ann_id
        print('img_id',img_id)
        image = torch.Tensor(self.images[img_id])
        print('image1',image)
        caption = self.captions[index]
        vocab = self.vocab

        ann_id = self.ids[index]
        img_id = ann_id[0]
        print('img_id2',img_id)
        image2 = torch.Tensor(self.images[img_id])
        print('image2',image2)
        path = self.dataset[img_id]['filename']
        image_raw = Image.open(os.path.join(self.newloc, path)).convert('RGB')
        if self.transform is not None:
            image_raw = self.transform(image_raw)

        # -------- The main difference between python2.7 and python3.6 --------#
        # The suggestion from Hongguang Zhu(https://github.com/KevinLight831)
        # ---------------------------------------------------------------------#
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        # convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, image_raw, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - image_raw: torch tensor of shape (3, 256, 256)
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - images_raw: torch tensor of shape (batch_size, 3, 256, 256)
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True) # target在第3个位置
    images, image_raw, captions, ids, img_ids = zip(*data)
    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)
    image_raw = torch.stack(image_raw, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, image_raw, targets, lengths, ids

def get_transform(data_name, data_split, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if data_split == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif data_split == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif data_split == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

def get_precomp_loader(data_path, data_split, vocab, transforms, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab, transforms)
    # for data in dset:
    #     print(data)
    #     break
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    transforms = get_transform(data_name, 'train', opt)
    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab, transforms,
                                      opt, batch_size, True, workers)

    transforms = get_transform(data_name, 'val', opt)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'val', vocab, transforms,
                                    opt, 100, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, transforms,
                                     opt, 100, False, workers)
    return test_loader
