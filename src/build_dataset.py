from torchvision import datasets
import argparse, os
import numpy as np
from glob import glob
import pickle
from sklearn.model_selection import train_test_split
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", default=1, type=int, help="random seed")
parser.add_argument("--dataset", "-d", default="svhn", type=str, help="dataset name : [svhn, cifar10,cifar100,miniImageNet,stl10]")


LABELS = {
            "mnist"         : [10,40,100,250,600,1000],
            "cifar10"       : [40,250,600,1000,2000,4000],
            "cifar10lt"     : [1000,2000,3000,4000],
            "cifar100"      : [2500,5000,10000],
            "stl10"         : [10,100,250,500,600,1000,2000,2500,4000],
            "svhn"          : [250,1000,2000,4000,5000],
            "mini_imagenet" : [250,1000,2000,4000,5000],
         }


_DATA_DIR         = "./dirs/data_storage"
_TORCH_VISION_DIR = 'torchvision'
_OFFLINE_DIR      = 'offline'
os.makedirs(os.path.join(_DATA_DIR,_TORCH_VISION_DIR),exist_ok=True)
os.makedirs(os.path.join(_DATA_DIR,_OFFLINE_DIR),exist_ok=True)


def gcn(images):
    images = images.astype(np.float) 
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    return images / per_image_norm

def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)

def get_img_num_per_cls(data, cls_num, imb_type, imb_factor):
    img_max = len(data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

def gen_imbalanced_data(data,targets,img_num_per_cls):
    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)

    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    new_targets = np.stack(new_targets)
    return (new_data, new_targets)

def verify_dataset(dataset_path):
    all_files = glob(os.path.join(dataset_path,'*.pkl'))
    for file in all_files:
        save_str = []
        with open(file, 'rb') as pkl_file:
            data = pickle.load(file=pkl_file)
            st   = 'file = {} images type = {} shape = {} labels type= {} shape = {}'.format(os.path.basename(file),type(data['images']),data['images'].shape,type(data['labels']),data['labels'].shape)
            save_str.append(st)
            print(st,flush=True)
            

def split_l_u(train_set, n_labels):
    rng                 = np.random.RandomState(args.seed)
    indices             = rng.permutation(len(train_set["images"]))
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices]
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // len(classes)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    u_labels2 = []
    for c in classes:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:]]
        u_labels += [np.zeros_like(c_labels[n_labels_per_cls:]) - 1] # dammy label
        u_labels2 += [c_labels[n_labels_per_cls:]] 
    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0),"labels2":np.concatenate(u_labels2, 0)}
    return l_train_set, u_train_set


def _load_cifar10():
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR10(os.path.join(_DATA_DIR,_TORCH_VISION_DIR,'cifar10'), train, download=True)
        data           = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.targets
        splits["train" if train else "test"] = data
    return splits.values()

def _load_cifar10LT():
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR10(os.path.join(_DATA_DIR,_TORCH_VISION_DIR,'cifar10'), train, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.targets
        if train:
            img_num_list = get_img_num_per_cls(data["images"],cls_num=10, imb_type='exp', imb_factor=0.1)
            data["images"],data["labels"] = gen_imbalanced_data(data["images"],data["labels"],img_num_list)
        splits["train" if train else "test"] = data    
    return img_num_list, splits.values()


def _load_svhn():
    splits = {}
    for split in ["train", "test",'extra']:
        tv_data = datasets.SVHN(os.path.join(_DATA_DIR,_TORCH_VISION_DIR,'svhn'), split, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.labels
        splits[split] = data
    return splits.values()


def _load_mnist():
    splits = {}
    for train in [True, False]:
        tv_data = datasets.MNIST(os.path.join(_DATA_DIR,_TORCH_VISION_DIR,'mnist'), train, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.targets
        splits["train" if train else "test"] = data
    return splits.values()


def _load_mini_imagenet():
    splits = {}    
    splits['train']= dict(np.load(os.path.join(_DATA_DIR,_OFFLINE_DIR,'mini_imagenet','train.npz')))
    splits['test'] = dict(np.load(os.path.join(_DATA_DIR,_OFFLINE_DIR,'mini_imagenet','test.npz')))
    return splits.values()

def _load_cifar100():
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR100(os.path.join(_DATA_DIR,_TORCH_VISION_DIR,'cifar100'), train, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.targets
        splits["train" if train else "test"] = data
    return splits.values()

def _load_stl10():
    splits = {}
    for split in ['train', 'test']:
        tv_data = datasets.STL10(os.path.join(_DATA_DIR,_TORCH_VISION_DIR,'stl10'), split=split, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.labels
        splits[split]  = data
    return splits.values()

def format(split_set):

    if isinstance(split_set['images'], torch.Tensor):
        split_set['images'] = split_set['images'].numpy()
    
    if isinstance(split_set['labels'], torch.Tensor):
        split_set['labels'] = split_set['labels'].numpy()
    elif isinstance(split_set['labels'], list):
        split_set['labels'] = np.array(split_set['labels'])

    if split_set['images'].ndim == 3:
        split_set['images'] = split_set['images'][:,:,:,None]
        split_set['images'] = split_set['images'].repeat(3,axis=-1)
    
    if  split_set['images'].shape[1] == 3:
        pass
    elif split_set['images'].shape[-1] == 3:
        split_set['images'] = np.transpose(split_set['images'],(0,3,1,2)).astype(np.uint8)

    split_set['labels']     = split_set['labels'].astype(np.int64)
    return split_set


def to_pickle(data_dict, dataset,name):
    with open(os.path.join(_DATA_DIR,dataset,'{}.pkl'.format(name)), 'wb') as pkl_file:
        pickle.dump(data_dict,file=pkl_file,protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args             = parser.parse_args()
    rng              = np.random.RandomState(args.seed)
    dataset_list = list(args.dataset.split(','))
    if 'all' in dataset_list:
        dataset_list = ['svhn','mnist','cifar10','cifar10lt','cifar100','mini_imagenet','stl10']

    for dataset in dataset_list:
        os.makedirs(os.path.join(_DATA_DIR,dataset),exist_ok=True)
        extra_set = None
        img_num_list = None

        if dataset   == "svhn":
            train_set, test_set,extra_set    = _load_svhn()
        elif dataset == "mnist":
            train_set, test_set              = _load_mnist()
        elif dataset == "cifar10":
            train_set,test_set               = _load_cifar10()
        elif dataset == "cifar10lt":
            img_num_list,(train_set,test_set)  = _load_cifar10LT()
        elif dataset == "cifar100":
            train_set,test_set               = _load_cifar100()
        elif dataset == "mini_imagenet":
            train_set, test_set              = _load_mini_imagenet()
        elif dataset == "stl10":
            train_set, test_set              = _load_stl10()
        
        train_set = format(train_set)
        test_set  = format(test_set)

        # permute index of training set
        indices             = rng.permutation(len(train_set["images"]))
        train_set["images"] = train_set["images"][indices]
        train_set["labels"] = train_set["labels"][indices]

        if extra_set is not None:
            extra_set           = format(extra_set)
            extra_indices       = rng.permutation(len(extra_set["images"]))
            extra_set["images"] = extra_set["images"][extra_indices]
            extra_set["labels"] = extra_set["labels"][extra_indices]

        X_train, X_val, y_train, y_val = train_test_split(train_set['images'],train_set['labels'],stratify=train_set['labels'], test_size=0.1)
        train_set                      = {"images": X_train, "labels": y_train}
        validation_set                 = {"images": X_val,   "labels": y_val}

        
        shapes = []
        shapes.append([f"train= {train_set['images'].shape[0]}"])
        shapes.append([f"val= {validation_set['images'].shape[0]}"])
        shapes.append([f"test= {test_set['images'].shape[0]}"])
        if extra_set is not None:
            shapes.append([f"extra= {extra_set['images'].shape[0]}"])
        shapes.append([f"size = {train_set['images'].shape[1:]}"])

        if img_num_list is not None:
            shapes.append([f"# images = {','.join(map(str,img_num_list))}"])
        np.savetxt(os.path.join(_DATA_DIR, dataset,'info.txt'),shapes,fmt='%s')

        for nlabels in LABELS[dataset]:
            print('creating dataset = {} nlabels = {} ....'.format(dataset,nlabels),flush=True)
            l_train_set, u_train_set  = split_l_u(train_set, nlabels)
            to_pickle(l_train_set,dataset, "l_train_{}".format(nlabels))
            to_pickle(u_train_set,dataset, "u_train_{}".format(nlabels))
            to_pickle(validation_set,dataset, "val_{}".format(nlabels))
            to_pickle(test_set,dataset,"test_{}".format(nlabels))
            if extra_set is not None:
                to_pickle(extra_set, dataset,"extra_{}".format(nlabels))
        verify_dataset(os.path.join(_DATA_DIR, dataset))