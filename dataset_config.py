# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import os
DATASET_ROOT_1 = '/media/data1/john/datasets/'
DATASET_ROOT_2 = '/media/data2/action_datasets/'
def return_ucf101(modality):
    filename_categories = 101
    root_data = os.path.join(DATASET_ROOT_1, 'ucf101')  # DATASET_ROOT_1
    annotation_data = os.path.join(root_data, 'file_list/ucf101')
    if modality == 'RGB':
        filename_imglist_train = os.path.join(annotation_data,'ucf101_train_split_1_rawframes.txt')
        filename_imglist_val = os.path.join(annotation_data,'ucf101_val_split_1_rawframes.txt')
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        filename_imglist_train = os.path.join(annotation_data,'ucf101_train_split_1_videos.txt*')
        filename_imglist_val = os.path.join(annotation_data,'ucf101_val_split_1_videos.txt*')
        prefix = 'frame{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    root_data = os.path.join(DATASET_ROOT_1, 'hmdb51')  # DATASET_ROOT_1
    annotation_data = os.path.join(root_data, 'file_list/hmdb51')

    if modality == 'RGB':
        filename_imglist_train = os.path.join(annotation_data,'hmdb51_train_split_1_rawframes.txt')
        filename_imglist_val = os.path.join(annotation_data,'hmdb51_val_split_1_rawframes.txt')
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        filename_imglist_train = os.path.join(annotation_data,'hmdb51_train_split_1_videos.txt')
        filename_imglist_val = os.path.join(annotation_data,'hmdb51_val_split_1_videos.txt')
        prefix = 'frame{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 174
    root_data = os.path.join(DATASET_ROOT_2,'sthv1')
    annotation_data = os.path.join(root_data,'annotations')
    if modality == 'RGB' or modality== 'RGBDiff':
        filename_imglist_train = os.path.join(annotation_data,'train_videofolder.txt')
        filename_imglist_val =  os.path.join(annotation_data,'val_videofolder.txt')
        prefix = '{:05d}.jpg'
    elif modality == 'Flow': # unused
        # root_data = DATASET_ROOT_1 + '/your_path_to/something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = '/your_path_to/something/v1/train_videofolder_flow.txt'
        filename_imglist_val = '/your_path_to/something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 174
    root_data = os.path.join(DATASET_ROOT_2, 'sthv2')
    annotation_data = os.path.join(root_data, 'annotations')
    if modality == 'RGB':
        filename_imglist_train = os.path.join(annotation_data,'train_videofolder.txt')
        filename_imglist_val = os.path.join(annotation_data,'val_videofolder.txt')
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow': # unused
        # root_data = DATASET_ROOT_1 + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = DATASET_ROOT_1 + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    root_data = os.path.join(DATASET_ROOT_2,'kinetics400')
    annotation_data = os.path.join(root_data,'annotations')



    if modality == 'RGB': # 输入的是帧文件
        filename_imglist_train = os.path.join(annotation_data,'train.txt')
        filename_imglist_val =  os.path.join(annotation_data,'val.txt')
        prefix = 'img_{:05d}.jpg'
    elif modality == 'VIDEO': # 输入的是视频文件
        pass # 暂时不处理
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51, 'sthv1':return_something, 'sthv2':return_somethingv2,
                   'kinetics': return_kinetics }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)

    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(DATASET_ROOT_1, file_imglist_train) # 暂时 DATASET_ROOT_1
    file_imglist_val = os.path.join(DATASET_ROOT_1, file_imglist_val) # 暂时
    if isinstance(file_categories, str):
        file_categories = os.path.join(DATASET_ROOT_1, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
