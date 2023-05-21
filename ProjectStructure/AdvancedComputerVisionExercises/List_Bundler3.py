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
        self.BA_list.append(q1)
        self.BA_list.append(q2)
        self.coord_3d_list.append(Q1)
        self.coord_3d_list.append(Q1)

    def append_keypoints(self, q1, q2, Q1, idx):
        self.last_stop += 1
        q2_to_list = [int(idx + 1), self.last_stop, q2[0], q2[1]]
        q1_to_list = [int(idx), self.last_stop, q1[0], q1[1]]
        Q1_to_list = [Q1[0], Q1[1], Q1[2]]
        self.append_instance(q2_to_list, q1_to_list, Q1_to_list)

    def duplicate_and_sort(self, temp_list):
        temp_list = sorted(temp_list, key=lambda x: (x[1], x[0]))
        match_counter = 0
        hit_counter = 0
        prev_frame_idx = 0
        cur_frame_idx = 1
        original_length = len(temp_list)
        old_unique_points = temp_list[-1][1]

        for i, x in enumerate(temp_list):
            if temp_list[i-1][0] == temp_list[i][0]:
                cur_frame_idx = i+1
                print("cur_frame_idx: " + str(cur_frame_idx))
                for i in range(cur_frame_idx-5, cur_frame_idx+5):
                    print(i, temp_list[i])
                break

        for i, x in enumerate(tqdm(temp_list, initial=cur_frame_idx, unit="sorting")):
            if temp_list[i][1] == temp_list[i - 1][1]: #chek if i is on q2
                if temp_list[i-1][0] == temp_list[i-2][0]: #checking for splits
                    prev_frame_idx = cur_frame_idx
                    cur_frame_idx = i



                for j in range(prev_frame_idx, i): # it goes from 2_frame_index to i
                    actual_frame = (temp_list[j][1] == temp_list[j - 1][1]) #check if j points to a q2 elemnt
                    correct_frame = (temp_list[j][0] == temp_list[i][0] - 1) #check if q2 element
                    if correct_frame and actual_frame:
                        same_x = (temp_list[i-1][2] == temp_list[j][2]) # check if q1 for i x is the same as q2 for j x
                        same_y = (temp_list[i-1][3] == temp_list[j][3]) #
                        if same_x and same_y:
                            #print("Match!")
                            #match_counter += 1
                            #print(match_counter)
                            temp_list[i][1] = temp_list[j][1]
                            temp_list[i - 1][1] = temp_list[j][1]
                            #del temp_list[j]
                            for k in range(i + 1, len(temp_list)): #Update uniqe point index.
                                temp_list[k][1] = temp_list[k][1] - 1

        temp_list = sorted(temp_list, key=lambda x: (x[1], x[0]))
        new_length = temp_list[-1][1]
        reduction = old_unique_points - new_length
        ratio = new_length/old_unique_points
        print("Old Size: " + str(old_unique_points) + ",  New Size: " + str(new_length))
        print("List Size Reduction: " + str(reduction))
        print("Ratio: " + str(ratio) + " times smaller")

        return temp_list

    def list_sort(self):
        temp_list = []
        for i in range(len(self.BA_list)):
            temp_list.append([self.BA_list[i][0], self.BA_list[i][1], self.BA_list[i][2], self.BA_list[i][3],
                         self.coord_3d_list[i][0], self.coord_3d_list[i][1], self.coord_3d_list[i][2]])

        temp_list = self.duplicate_and_sort(temp_list)
        print("# of unique points: {}".format(temp_list[-1][1]))
        #temp_list = self.remove_frame_zero(temp_list)

        self.BA_list = []
        self.coord_3d_list = []
        for i in range(len(temp_list)):
            self.BA_list.append([temp_list[i][0], temp_list[i][1], temp_list[i][2], temp_list[i][3]])
            self.coord_3d_list.append([temp_list[i][4], temp_list[i][5], temp_list[i][6]])

    def remove_frame_zero(self, temp_list):
        temp_list = sorted(temp_list, key=lambda x: (x[0], x[1]))
        i = 0
        while i < len(temp_list):
            if temp_list[i][0] == 0:
                del temp_list[i]
            else:
                i += 1
        temp_list = sorted(temp_list, key=lambda x: (x[1], x[0]))
        return temp_list
