from torchvision import datasets
import argparse, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_rootdir", default='', type=str, help="folder to store the data")
parser.add_argument("--seed", "-s", default=1, type=int, help="random seed")
parser.add_argument("--dataset", "-d", default="svhn", type=str, help="dataset name : [svhn, cifar10]")
parser.add_argument("--nlabels", "-n", default=1000, type=int, help="the number of labeled data")
parser.add_argument("--normalization", default='unnormalized_HWC', type=str, help="[gcn_zca_normalized_CHW, unnormalized_HWC]")

#confirmed: same as Oliver
COUNTS = {
    "svhn": {"train": 73257, "test": 26032, "valid": 7326, "extra": 531131},
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0},
    "imagenet_32": {
        "train": 1281167,
        "test": 50000,
        "valid": 50050,
        "extra": 0,
    },
}


#confirmed: this is different from Oliver setting, but this is more resonable since it ensure no same example present in both labeled and unlabeled. 
def split_l_u(train_set, n_labels):
    # NOTE: this function assume that the train_set is shuffled.
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // len(classes)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    for c in classes:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:]]
        u_labels += [np.zeros_like(c_labels[n_labels_per_cls:]) - 1] # dammy label
    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}
    return l_train_set, u_train_set


#confirmed with pytorch.org and several other repos: this is the correct way to download with torchvision dataset
def _load_svhn(data_rootdir):
    splits = {}
    for split in ["train", "test", "extra"]:
        tv_data = datasets.SVHN(data_rootdir, split, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.labels
        splits[split] = data
    return splits.values()


#confirmed with pytorch.org and several other repos: this is the correct way to download with torchvision dataset
def _load_cifar10(data_rootdir):
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR10(data_rootdir, train, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = np.array(tv_data.targets)
        splits["train" if train else "test"] = data
    return splits.values()




if __name__=='__main__':
    args = parser.parse_args()
    

    #confirmed: same as Oliver;
    #random seed for separating train and val set
    rng = np.random.RandomState(args.seed)

    #confirmed: same as Oliver;
    validation_count = COUNTS[args.dataset]["valid"]

    #confirmed: same as Oliver;
    extra_set = None  # In general, there won't be extra data.
    if args.dataset == "svhn":
        train_set, test_set, extra_set = _load_svhn(args.data_rootdir)
    elif args.dataset == "cifar10":
        train_set, test_set = _load_cifar10(args.data_rootdir)
        

    #confirmed: same as Oliver;
    # permute index of training set
    indices = rng.permutation(len(train_set["images"]))
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices]

    #confirmed: same as Oliver;
    if extra_set is not None:
        extra_indices = rng.permutation(len(extra_set["images"]))
        extra_set["images"] = extra_set["images"][extra_indices]
        extra_set["labels"] = extra_set["labels"][extra_indices]

    #confirmed: same as Oliver;
    # split training set into training and validation
    train_images = train_set["images"][validation_count:]
    train_labels = train_set["labels"][validation_count:]
    validation_images = train_set["images"][:validation_count]
    validation_labels = train_set["labels"][:validation_count]
    validation_set = {"images": validation_images, "labels": validation_labels}
    train_set = {"images": train_images, "labels": train_labels}


    # split training set into labeled data and unlabeled data
    l_train_set, u_train_set = split_l_u(train_set, args.nlabels)

    if not os.path.exists(os.path.join(args.data_rootdir, 'Regular_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels))):
        os.makedirs(os.path.join(args.data_rootdir, 'Regular_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels)))

    np.save(os.path.join(args.data_rootdir, 'Regular_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), "l_train"), l_train_set)
    np.save(os.path.join(args.data_rootdir, 'Regular_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), "u_train"), u_train_set)
    np.save(os.path.join(args.data_rootdir, 'Regular_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), "val"), validation_set)
    np.save(os.path.join(args.data_rootdir, 'Regular_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), "test"), test_set)
    if extra_set is not None:
        np.save(os.path.join(args.data_rootdir, 'Regular_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), "extra"), extra_set)




