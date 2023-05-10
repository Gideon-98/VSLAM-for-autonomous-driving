import numpy as np
from tqdm import tqdm

import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# BoW code inspired by https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb


class ListBundler:

    def __init__(self, n_clusters=200, n_features=500, future_iterations=6):
        self.BA_list = []
        self.coord_3d_list = []
        self.last_stop = -1

    def append_instance(self, q2, q1, Q1):
        self.BA_list.append(q2)
        self.BA_list.append(q1)
        self.coord_3d_list.append(Q1)
        self.coord_3d_list.append(Q1)

    def append_keypoints(self, q1, q2, Q1, idx):
        self.last_stop += 1
        q2_to_list = [int(idx - 1), self.last_stop, q2[0], q2[1]]
        q1_to_list = [int(idx), self.last_stop, q1[0], q1[1]]
        Q1_to_list = [Q1[0], Q1[1], Q1[2]]
        self.append_instance(q2_to_list, q1_to_list, Q1_to_list)

    def duplicate_sort(self, temp_list):
        prev_frame_idx = 0
        for i, x in enumerate(tqdm(temp_list, unit="sorting")):

            if temp_list[i][0] is not (temp_list[prev_frame_idx][0] + 1):
                prev_frame_idx = (i - 1)

            for j in range(prev_frame_idx, i):
                correct_frame = (temp_list[i][0] == (temp_list[j][0] - 1))
                same_x = (temp_list[i][2] == temp_list[j][2])
                same_y = (temp_list[i][3] == temp_list[j][3])
                if correct_frame and same_x and same_y:
                    temp = temp_list[i]
                    del temp_list[i]
                    temp[1] = temp_list[j][1]
                    temp_list.insert((j + 1), temp)

        return temp_list

    def list_sort(self):
        temp_list = []
        for i in range(len(self.BA_list)):
            temp_list.append([self.BA_list[i][0], self.BA_list[i][1], self.BA_list[i][2], self.BA_list[i][3],
                         self.coord_3d_list[i][0], self.coord_3d_list[i][1], self.coord_3d_list[i][2]])

        temp_list = sorted(temp_list, key=lambda x: (x[1], x[0]))
        temp_list = self.duplicate_sort(temp_list)
        temp_list = sorted(temp_list, key=lambda x: (x[1], x[0]))

        self.BA_list = []
        self.coord_3d_list = []
        for i in range(len(temp_list)):
            self.BA_list.append([temp_list[i][0], temp_list[i][1], temp_list[i][2], temp_list[i][3]])
            self.coord_3d_list.append([temp_list[i][4], temp_list[i][5], temp_list[i][6]])