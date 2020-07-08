from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import cv2

class ObjLoader:
    def __init__(self):
        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []

        self.vertex_index = []
        self.texture_index = []
        self.normal_index = []

        self.model = []

    def load_model(self, file):
        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                self.vert_coords.append(values[1:4])
            if values[0] == 'vt':
                self.text_coords.append(values[1:3])
            if values[0] == 'vn':
                self.norm_coords.append(values[1:4])

            if values[0] == 'f':
                face_i = []
                text_i = []
                norm_i = []
                count = 0
                for value in values[1:4]:
                    args = value.split('/')
                    if(args[0] != '' and count == 0):
                        face_i.append(int(args[0])-1)
                    if(args[1] != ''):
                        text_i.append(int(args[1])-1)
                    if(args[2] != ''):
                        norm_i.append(int(args[2])-1)
                    count = count + 1
                self.vertex_index.append(face_i)
                self.texture_index.append(text_i)
                self.normal_index.append(norm_i)

        self.vertex_index = [y for x in self.vertex_index for y in x]
        self.texture_index = [y for x in self.texture_index for y in x]
        self.normal_index = [y for x in self.normal_index for y in x]

        for i in self.vertex_index:
            self.model.extend(self.vert_coords[i])

        for i in self.texture_index:
            self.model.extend(self.text_coords[i])

        for i in self.normal_index:
            self.model.extend(self.norm_coords[i])

        self.model = np.array(self.model, dtype='float32')

if __name__ == "__main__":
    obj_head = ObjLoader()
    obj_head.load_model("male_head.obj")
    
    head_pts = [[vert[0], vert[1]] for vert in obj_head.vert_coords]
    head_pts = np.array(head_pts, np.float32)
    
    obj_landmarks = ObjLoader()
    obj_landmarks.load_model("male_head_landmarks.obj")

    landmarks_ref = [[vert[0], vert[1]] for vert in obj_landmarks.vert_coords]
    landmarks_ref = np.array(landmarks_ref, np.float32)
    
    head_pts = head_pts - landmarks_ref[29]  # Nose tip is index 29
    head_pts_x, head_pts_y = zip(*head_pts)
    plt.scatter(head_pts_x, head_pts_y)

    landmarks_ref = landmarks_ref - landmarks_ref[29]  # Nose tip is index 29
    landmarks_ref_x, landmarks_ref_y = zip(*landmarks_ref)
    plt.scatter(landmarks_ref_x, landmarks_ref_y, color='r')
    
    jaw = [50, 56, 55, 67, 54, 53, 52, 51, 33, 14, 15, 16, 17, 28, 18, 19, 13]
    right_eyebrow = [41,40,39,38,37] # his right
    left_eyebrow = [0,1,2,3,4]
    nose = [34,36,35,29,30,43,42,30,5,6]
    right_eye = [60,59,58,57,62,61] # his right
    left_eye = [20,21,22,23,24,25]
    mouth = [47,45,44,31,7,8,10,11,12,64,49,48,46,65,32,26,9,27,63,66]
    #inner_mouth=[]

    for i, pt in enumerate(landmarks_ref):
        plt.annotate(i, (landmarks_ref_x[i], landmarks_ref_y[i]))

    plt.show()
