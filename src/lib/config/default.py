from lib import datasets

dataset_config= {
    'cifar10'   : { "dataset": datasets.CIFAR10,     "num_classes" : 10  ,"transform"   : [True, True, True], "model": 'wr28-2' },
    'cifar10lt' : { "dataset": datasets.CIFAR10lt,   "num_classes" : 10  ,"transform"   : [True, True, True], "model": 'wr28-2' },
    'stl10'     : { "dataset": datasets.STL10,       "num_classes" : 10  ,"transform"   : [True, True, True], "model": 'wr28-2' },
    'mnist'     : { "dataset": datasets.MNIST,       "num_classes" : 10  ,"transform"   : [True, True, True], "model": 'wr28-2' },
    'svhn'      : { "dataset": datasets.SVHN,        "num_classes" : 10  ,"transform"   : [True, True, True], "model": 'wr28-2' },
    'cifar100'  : { "dataset": datasets.CIFAR100,    "num_classes" : 100 ,"transform"   : [True, True, True], "model": 'wr28-8' },
    'mimagenet' : { "dataset": datasets.MImagenet,   "num_classes" : 100 ,"transform"   : [True, True, True], "model": 'wr28-8' },
}

shared_config = {
    "iteration"       : 500000,
    "warmup"          : 200000,
    "lr_decay_iter"   : 400000,
    "lr_decay_factor" : 0.2,
    "batch_size"      : 100,
    "T"               : 1,
    "set_cenmode"     : "super",
    "lr"              : -3,
}


alg_config= {
    'NONE': {'alg':'NONE'}, 
    'MT'  : {'alg':'MT','ema_factor':0.95},
    'VAT' : {'alg':'VAT',"xi" : 1e-6,"eps": 6},
    'PI'  : {'alg':'PI'},
    'PL'  : {'alg':'PL',"threashold" : 0.95},
}


ce_config = {
    'NONE': {'gamma':'-inf','beta':'-inf'}, 
    'MT'  : {'gamma':-0.81 ,'beta':'-inf'},
    'VAT' : {'gamma':0.75  ,'beta':'-inf'},
    'PI'  : {'gamma':-0.70 ,'beta':'-inf'},
    'PL'  : {'gamma':-0.73 ,'beta':'-inf'},
}


cecm_config = {
    'NONE': {'gamma':'-inf','beta':-0.73}, 
    'MT'  : {'gamma':-0.81 ,'beta':-0.89 },
    'VAT' : {'gamma':0.75  ,'beta':-0.20 },
    'PI'  : {'gamma':-0.70 ,'beta':-0.75 },
    'PL'  : {'gamma':-0.73 ,'beta': 0.48 },
}



labels_config = {k:{'nlabels':k} for k in [10,40,100,250,500,600,1000,2000,2500,3000,4000,5000]}
