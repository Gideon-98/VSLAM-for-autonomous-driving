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

    def duplicate_and_sort_no_removal(self, temp_list):
        temp_list = sorted(temp_list, key=lambda x: (x[1], x[0]))
        prev_frame_idx = 0
        cur_frame_idx = 1
        old_unique_points = temp_list[-1][1]

        for i in range(3, len(temp_list), 2):
            if temp_list[i - 1][0] == temp_list[i - 2][0]:
                cur_frame_idx = i
                break

        with tqdm(total=len(temp_list) - 1, unit="Sorting") as pbar:
            i = cur_frame_idx
            while i < len(temp_list):
                if temp_list[i - 1][0] == temp_list[i - 2][0]:
                    prev_frame_idx = cur_frame_idx
                    cur_frame_idx = i

                for j in range(prev_frame_idx, i, 2):  # it goes from 2_frame_index to i
                    same_x = (temp_list[i - 1][2] == temp_list[j][2])  # check if q1 for i x is the same as q2 for j x
                    same_y = (temp_list[i - 1][3] == temp_list[j][3])  #
                    if same_x and same_y:
                        temp_list[i][1] = temp_list[j][1]
                        temp_list[i - 1][1] = temp_list[j][1]
                        temp_list[i][4] = temp_list[j][4]
                        temp_list[i - 1][4] = temp_list[j][4]
                        temp_list[i][5] = temp_list[j][5]
                        temp_list[i - 1][5] = temp_list[j][5]
                        temp_list[i][6] = temp_list[j][6]
                        temp_list[i - 1][6] = temp_list[j][6]
                        # del temp_list[j]
                        for k in range(i + 1, len(temp_list)):  # Update uniqe point index.
                            temp_list[k][1] = temp_list[k][1] - 1
                        del temp_list[i - 1]
                        i += 1
                        break
                else:
                    del temp_list[i - 1]
                    del temp_list[i - 1]
                pbar.update(2)
        temp_list = sorted(temp_list, key=lambda x: (x[1], x[0]))
        new_length = temp_list[-1][1]
        reduction = old_unique_points - new_length
        ratio = new_length / old_unique_points
        print("Old Size: " + str(old_unique_points) + ",  New Size: " + str(new_length))
        print("List Size Reduction: " + str(reduction))
        print("Ratio: " + str(ratio) + " times smaller \n")
        return temp_list

    def duplicate_and_sort(self, temp_list):

        temp_list = sorted(temp_list, key=lambda x: (x[1], x[0]))  # Sort the list, with uniqe point being first then frames.
        prev_frame_idx = 0
        cur_frame_idx = 1
        old_unique_points = temp_list[-1][1]
        newtemp = []

        for i, x in enumerate(temp_list):  # Function checks if we are working with keypoints from frame 1
            if temp_list[i - 1][0] == temp_list[i][0]:  # if we go from one frame of data to another
                cur_frame_idx = i  # Points to the first q1 of frame 1 # Before it was i+1, since lucases code was looking at the first q2, not the first q1
                break
        with tqdm(total=len(temp_list) - 1, unit="Sorting") as pbar:
            for i in range(cur_frame_idx + 1, len(temp_list), 2):

                actual_frame = (temp_list[i][1] == temp_list[i - 1][1])  # check if j points to a q2 elemnt
                correct_frame = (temp_list[i][0] - 1 == temp_list[i - 1][0])  # check if q2 element
                if actual_frame == False and correct_frame == False:
                    print("Oh NO!")
                if actual_frame and correct_frame:

                    if temp_list[i][0] != temp_list[i - 2][0] and cur_frame_idx != i:  # Check if the trackpoint we are looking at is the first of it's related frame. Happens first when we go from

                        prev_frame_idx = cur_frame_idx  # Update what "keyframe" we are looking at
                        cur_frame_idx = i

                    for k in range(prev_frame_idx + 1, i, 2):

                        same_x = (temp_list[i][2] == temp_list[k][2])  # check if q1 for i x is the same as q2 for j x
                        same_y = (temp_list[i][3] == temp_list[k][3])  #

                        if same_x and same_y and actual_frame and correct_frame:

                            newtemp.append(temp_list[k-1])
                            newtemp.append(temp_list[k])
                            newtemp.append([temp_list[i][0], temp_list[k][1], temp_list[i][2], temp_list[i][3],
                                            temp_list[k][4], temp_list[k][5], temp_list[k][6]])
                            newtemp.append([temp_list[i-1][0], temp_list[k][1], temp_list[i-1][2], temp_list[i-1][3],
                                            temp_list[k][4], temp_list[k][5], temp_list[k][6]])

                pbar.update(2)

        for i in range(0, len(newtemp), 4):
            newtemp[i][1] = int(i / 4)
            newtemp[i + 1][1] = int(i / 4)
            newtemp[i + 2][1] = int(i / 4)
            newtemp[i + 3][1] = int(i / 4)

        temp_list = sorted(newtemp, key=lambda x: (x[1], x[0]))
        new_length = temp_list[-1][1]
        reduction = old_unique_points - new_length
        ratio = new_length / old_unique_points
        print("Old Size: " + str(old_unique_points) + ",  New Size: " + str(new_length))
        print("List Size Reduction: " + str(reduction))
        print("Ratio: " + str(ratio) + " times smaller \n")
        return temp_list

    def list_sort(self):
        temp_list = []
        for i in range(len(self.BA_list)):
            temp_list.append([self.BA_list[i][0], self.BA_list[i][1], self.BA_list[i][2], self.BA_list[i][3],
                              self.coord_3d_list[i][0], self.coord_3d_list[i][1], self.coord_3d_list[i][2]])

        temp_list = self.duplicate_and_sort_no_removal(temp_list)

        print("# of unique points: {}".format(temp_list[-1][1]))
        # temp_list = self.remove_frame_zero(temp_list)

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