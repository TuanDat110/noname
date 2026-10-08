import numpy as np
import pickle
from torch.utils.data import Dataset
import random

from feeders import tools

# Joint permutation that mirrors the skeleton left <-> right.
flip_index = np.concatenate(
    (
        [0, 2, 1, 4, 3, 6, 5],
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ),
    axis=0,
)

# ============================================================
# Joint partition. Original 27 joints:
#   0-6:   upper body (7 joints)
#   7-16:  right hand  (10 joints: wrist + 9 fingers)
#   17-26: left hand   (10 joints: wrist + 9 fingers)
#
# Reordered into 3 groups of 9:
#   Group 0: upper body + both wrists  [0,1,2,3,4,5,6, 7,17]
#   Group 1: right hand fingers         [8,9,10,11,12,13,14,15,16]
#   Group 2: left hand fingers          [18,19,20,21,22,23,24,25,26]
# ============================================================
PARTITION_IDX = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 17,        # group 0: upper body + wrists
    8, 9, 10, 11, 12, 13, 14, 15, 16,   # group 1: right hand fingers
    18, 19, 20, 21, 22, 23, 24, 25, 26,  # group 2: left hand fingers
])


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=True,
        random_mirror=False,
        random_mirror_p=0.5,
        is_vector=False,
        bone_stream=False,
        motion_stream=False,
        num_class=2000,
        partition=True,
    ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.load_data()
        self.is_vector = is_vector
        self.bone_stream = bone_stream
        self.motion_stream = motion_stream
        self.num_class = num_class
        self.partition = partition
        if normalization:
            self.get_mean_map()

        print(len(self.label))

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            with open(self.label_path, "rb") as f:
                self.sample_name, self.label = pickle.load(f, encoding="latin1")

        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode="r")
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 4, 1, 3))
            .reshape((N * T * M, C * V))
            .std(axis=0)
            .reshape((C, 1, V, 1))
        )

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]  # C T V M
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        data_numpy[np.isinf(data_numpy)] = 0

        if self.bone_stream:
            ori_data = data_numpy
            for v1, v2 in (
                (5, 6), (5, 7), (6, 8), (8, 10), (7, 9), (9, 11),
                (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
                (14, 15), (16, 17), (18, 19), (20, 21),
                (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
                (24, 25), (26, 27), (28, 29), (30, 31),
                (10, 12), (11, 22),
            ):
                data_numpy[:, :, v2 - 5, :] = (
                    ori_data[:, :, v2 - 5, :] - ori_data[:, :, v1 - 5, :]
                )

        if self.motion_stream:
            T = data_numpy.shape[1]
            ori_data = data_numpy
            for t in range(T - 1):
                data_numpy[:, t, :, :] = ori_data[:, t + 1, :, :] - ori_data[:, t, :, :]
            data_numpy[:, T - 1, :, :] = 0

        if self.random_choose:
            data_numpy = tools.random_sample_np(data_numpy, self.window_size)
        else:
            data_numpy = tools.uniform_sample_np(data_numpy, self.window_size)

        if self.random_mirror:
            if random.random() > self.random_mirror_p:
                assert data_numpy.shape[2] == 27
                data_numpy = data_numpy[:, :, flip_index, :]
                if self.is_vector:
                    data_numpy[0, :, :, :] = -data_numpy[0, :, :, :]
                else:
                    data_numpy[0, :, :, :] = 512 - data_numpy[0, :, :, :]

        if self.normalization:
            assert data_numpy.shape[0] == 3
            if self.is_vector:
                data_numpy[0, :, 0, :] = data_numpy[0, :, 0, :] - data_numpy[
                    0, :, 0, 0
                ].mean(axis=0)
                data_numpy[1, :, 0, :] = data_numpy[1, :, 0, :] - data_numpy[
                    1, :, 0, 0
                ].mean(axis=0)
            else:
                data_numpy[0, :, :, :] = data_numpy[0, :, :, :] - data_numpy[
                    0, :, 0, 0
                ].mean(axis=0)
                data_numpy[1, :, :, :] = data_numpy[1, :, :, :] - data_numpy[
                    1, :, 0, 0
                ].mean(axis=0)

        if self.random_shift:
            if not self.bone_stream:
                if self.is_vector:
                    data_numpy[0, :, 0, :] += random.random() * 20 - 10.0
                    data_numpy[1, :, 0, :] += random.random() * 20 - 10.0
                else:
                    data_numpy[0, :, :, :] += random.random() * 20 - 10.0
                    data_numpy[1, :, :, :] += random.random() * 20 - 10.0

        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # Joint reordering. Must happen after all augmentations (mirror, bone, ...).
        if self.partition:
            data_numpy = data_numpy[:, :, PARTITION_IDX, :]

        # Frame indices [0, 1, ..., window_size-1] for the temporal embedding.
        index_t = np.arange(self.window_size, dtype=np.float32)

        # data_numpy [C, T, V, M], index_t [T], label, sample index in the dataset
        return data_numpy, index_t, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def per_class_acc_top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [
            l in rank[i, -top_k:] for i, l in enumerate(self.label)
        ]
        labels = np.asarray(self.label, dtype=np.int64)
        count = np.bincount(labels, minlength=self.num_class)
        hit = np.bincount(labels, weights=np.asarray(hit_top_k, dtype=np.float64),
                          minlength=self.num_class)
        # Average over the classes that occur in this split; a class without
        # samples has no accuracy and would otherwise turn the mean into NaN.
        present = count > 0
        return np.mean(hit[present] / count[present])
