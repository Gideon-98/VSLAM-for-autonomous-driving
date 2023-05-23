from tqdm import tqdm

class ListBundler:
    def __init__(self, n_clusters=200, n_features=500, future_iterations=6):
        self.BA_list = []
        self.coord_3d_list = []
        self.last_stop = -1
        self.keypoints = []
        self.saved_orig = []
        self.saved_track = []

        self.bigTemp = []
        self.iList = []

    def save_tracks(self, tp_list_orig, tp_list_track):
        self.saved_orig.append(tp_list_orig)
        self.saved_track.append(tp_list_track)

    def pre_sorter(self):


    def append_3d(self):
        last_stop = 0
        for i in range(len(self.iList)):
            for j in range(len(self.iList[i])):
                for k in range(len(self.iList[i][j])):
                    self.iList[i][j][k].append(self.coord_3d_list[last_stop][0])
                    self.iList[i][j][k].append(self.coord_3d_list[last_stop][1])
                    self.iList[i][j][k].append(self.coord_3d_list[last_stop][2])
                    self.iList[i][j][k].append(last_stop)
                    last_stop += 1
        '''
        Form after this: self.iList[frame_idx][j][k] = [2dx, 2dy, 3dx, 3dy, 3dz, Q_idx]
        '''
    def big_sorter(self):
        self.pre_sorter() "Turn everytning into the pic I sent"
        self.append_3d()                "Adds 3D points to each tp"
        '''
        Form after this: self.iList[frame_idx][j][k] = [2dx, 2dy, 3dx, 3dy, 3dz, Q_idx]
        '''

        for i in range(len(self.iList[0])):
            self.bigTemp.append([0, i, self.iList[0][i][0], self.iList[0][i][1],
                                 self.coord_3d_list[i][0], self.coord_3d_list[i][1], self.coord_3d_list[i][2]])

        "Look for pixel matches in previous trackpoints"

        for i in enumerate(tqdm(self.iList, initial=1, unit="sorting")):
            for j in range(len(self.iList[i])):
                for k in range(1, len(self.iList[i][j])):
                    its = self.future_iterations
                    if i < its:
                        its = i
                    for l in range(its):
                        for m in range(len(self.iList[i-l][j])):
                            match_x = (self.iList[i-l][m][k-l][0] == self.iList[i][j][k][0])
                            match_y = (self.iList[i-l][m][k-l][1] == self.iList[i][j][k][1])
                            if match_x and match_y:
                                self.iList[i][j][k][5] = self.iList[i-l][m][k-l][5]

    def bundler(self):
        self.big_sorter()
        tmp_BA = []
        tmp_3D = []
        tmp_full =[]
        for i in range(1, len(self.iList)):
            for j in range(len(self.iList[i])):
                for k in range(1, len(self.iList[i][j])):
                    tmp_full.append([i, self.iList[i][j][k][5], self.iList[i][j][k][0], self.iList[i][j][k][1],
                                     self.iList[i][j][k][2], self.iList[i][j][k][3], self.iList[i][j][k][4]])

        tmp_full = sorted(tmp_full, key=lambda x: (x[1], x[0]))
        for i in range(len(tmp_full)):
            tmp_BA.append([tmp_full[i][0], tmp_full[i][1], tmp_full[i][2], tmp_full[i][3]])
            tmp_3D.append([tmp_full[i][4], tmp_full[i][5], tmp_full[i][6]])

        self.BA_list = tmp_BA
        self.coord_3d_list = tmp_3D