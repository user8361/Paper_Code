# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 14:33:59
import os
import cv2

import numpy as np
from numpy.random import randint

import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from numba import jit #,cuda,roc,vectorize
import warnings

warnings.filterwarnings('ignore')



# 使用numba对python代码进行加速（支持numpy库，其他...）
#@cuda.jit(device = True) # 报错（ 'DeviceFunctionTemplate' object is not callable）
@jit(nopython=True, nogil=True, cache=True)
def get_momentums(height, width, lst):
    change = 0
    for h in range(1, height):
        for w in range(1, width):
            if lst[h][w] > 0:
                change += 1
    return change

@jit(nopython=True, nogil=True, cache=True)
def get_idx(img_change_rate,v):
    '''

    :param img_change_list: 帧变化率列表
    :param v: 阈值，当变化率大于这个阈值才将索引返回
    :return:
    '''
    idx_list = []
    for idx, rate in enumerate(img_change_rate):
        if rate > v:  # 动量变化率大于0.2时，做一次记录
            idx_list.append(idx)

    return idx_list
class VideoRecord(object):
    """Store the basic information of the video

    _data[0]: the absolute path of the video frame folder
    _data[1]: the frame number
    _data[2]: the label of the video
    """

    def __init__(self, row):
        self._data = row

    @property
    def path(self):

        return os.path.join('/media/data1/john/datasets/hmdb51/rawframes/',self._data[0])

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    """The torch dataset for the video data.

    :param list_file: the list file is utilized to specify the data sources.
    Each line of the list file contains a tuple of extracted video frame folder path (absolute path),
    video frame number, and video groundtruth class. An example line looks like:
    /data/xxx/xxx/Dataset/something-somthing-v1/100218 42 134
    帧文件路径 ，  视频帧数量  ，  视频的ground truth
    """

    def __init__(
            self, list_file, num_segments=8, new_length=1, modality='RGB',
            image_tmpl='img_{:05d}.jpg', transform=None, random_shift=True,
            test_mode=False, remove_missing=False, multi_clip_test=False,
            dense_sample=False):
        
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.multi_clip_test = multi_clip_test
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.root_path = '/media/data1/john/datasets/hmdb51/rawframes/'
        print("=> Preparing data ...")
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:

                return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(directory, self.image_tmpl.format(1))).convert('RGB')]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]


        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """Random Sampling from each video segment

        :param record: VideoRecord
        :return: list
        """

        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            # average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            # # print('dataset 106 ==> average_duration : ',average_duration)
            # if average_duration > 0:
            #     # np.multiply 矩阵对应元素乘
            #     offsets = np.multiply(list(range(self.num_segments)), average_duration)\
            #               + randint(average_duration, size=self.num_segments)
            #     # print('dataset 111 ==> offsets+1 : ',offsets+1)
            # elif record.num_frames > self.num_segments:
            #     offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            # else:
            #     offsets = np.zeros((self.num_segments,))
            # return offsets + 1 # 列表中的值分别加1


            # 基于TSN 稀疏采样
            #   根据帧差进行筛选帧（将变化率较小的帧剔除）
            #   改进：将大概率的帧进行保存（将帧进行分组（n_segments），每个segment中抽取一帧）
            num_frames = record.num_frames  # 帧的总数量
            dir_frames = os.path.join(self.root_path, record.path)  # 帧的父目录
            img_list = os.listdir(dir_frames)  # 获得文件夹中的所有图片文件
            if self.image_tmpl =='img_{:05d}.jpg':
                img_list.sort(key=lambda x: int(x[5:-4]))  # 进行排序 针对hmdb51和ucf101数据集  pref = img_{:05d}.jpg
            else:
                img_list.sort(key=lambda x: int(x[0:-4]))  # 进行排序 针对something数据集  pref = {:05d}.jpg
            img_gray_list = []  # 灰度图列表
            img_change_rate = []  # 图片动量变化率
            tmp_img = cv2.imread(os.path.join(dir_frames, img_list[0]))  # 获得文件夹中的第一张图像用来求尺寸
            width = tmp_img.shape[1]  # 宽
            height = tmp_img.shape[0]  # 高

            for i in img_list:
                img = cv2.imread(os.path.join(dir_frames, i))
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 将图像转成Gray
                img_gray_list.append(img_gray)

            # 求三帧图像的帧差结果
            for i in range(0, num_frames - 2):
                diff1 = cv2.absdiff(img_gray_list[i], img_gray_list[i + 1])  #
                diff2 = cv2.absdiff(img_gray_list[i + 2], img_gray_list[i + 1])
                diff = cv2.bitwise_and(diff1, diff2)
                # 动量，计算动量的变化率，来选择要采样的帧
                change = get_momentums(height, width, diff)
                img_change_rate.append(float(change) / (width * height))  # 动量的变化率

            idx_list = get_idx(img_change_rate, v=0.2)
            len_idx_list = len(idx_list)

            if len_idx_list > self.num_segments:
                average_duration = (len_idx_list - self.new_length + 1) // self.num_segments
                offsets = []
                if average_duration > 0:
                    offsets_tmp = np.multiply(list(range(self.num_segments)), average_duration) + randint(
                        average_duration,
                        size=self.num_segments)
                else:
                    offsets_tmp = np.zeros((self.num_segments,))
                for i in offsets_tmp:
                    offsets.append(np.array(idx_list[i], dtype=int))
            elif len_idx_list == self.num_segments:
                offsets = np.array(idx_list, dtype=int)

            else:
                average_duration = (record.num_frames - self.new_length + 1) // self.num_segments

                # 采样间隔
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                      size=self.num_segments)
                elif record.num_frames > self.num_segments:
                    offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
                else:
                    offsets = np.zeros((self.num_segments,))
            offsets_tmp = []
            for tmp in offsets:
                offsets_tmp.append(tmp + 1)
            return offsets_tmp

    def _get_val_indices(self, record):
        """Sampling for validation set

        Sample the middle frame from each video segment
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        record = self.video_list[index]
        # check this is a legit video folder
        file_name = self.image_tmpl.format(1)
        full_path = os.path.join(record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(record.path, file_name))

            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(record.path, file_name)

        if not self.test_mode:  # training or validation set
            if self.random_shift:  # training set
                segment_indices = self._sample_indices(record)
            else:  # validation set
                segment_indices = self._get_val_indices(record)
        else:  # test set
            # for mulitple clip test, use random sampling;
            # for single clip test, use middle sampling
            if self.multi_clip_test:
                segment_indices = self._sample_indices(record)
            else:
                segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                # import pdb
                # pdb.set_trace()
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

# if __name__ == "__main__":
#     # test dataset
#     test_train_list = '/media/data1/john/datasets/ucf101/file_list/ucf101/ucf101_train_split_1_rawframes.txt'
#     test_num_segments = 8
#     data_length = 1
#     test_modality = 'RGB'
#     prefix = 'img_{:05d}.jpg'
#
#     from transforms import get_transforms
#
#     train_dataset = TSNDataSet(
#         test_train_list, num_segments=test_num_segments,
#         new_length=data_length, modality=test_modality,
#         image_tmpl=prefix, multi_clip_test=False, dense_sample=False,
#         transform=get_transforms()
#
#     )
#     data, label = train_dataset.__getitem__(3)

