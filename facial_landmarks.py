# USAGE
# python3 facial_landmarks.py --image images/kevin.jpg 

# import the necessary packages
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import math

from face_model import ObjLoader

# FROM: https://stackoverflow.com/questions/8457645/efficient-pythonic-way-to-snap-a-value-to-some-grid
import bisect
def snap(myGrid, myValue):
	ix = bisect.bisect_right(myGrid, myValue[0])
	if ix == 0:
		return myGrid[0]
	elif ix == len(myGrid):
		return myGrid[-1]
	else:
		return min(myGrid[ix - 1], myGrid[ix], key=lambda gridValue: abs(gridValue - myValue))


def ndsnap(points, grid):
    """
    Snap an 2D-array of points to values along an 2D-array grid.
    Each point will be snapped to the grid value with the smallest
    city-block distance.

    Parameters
    ---------
    points: 2D-array. Must have same number of columns as grid
    grid: 2D-array. Must have same number of columns as points

    Returns
    -------
    A 2D-array with one row per row of points. Each i-th row will
    correspond to row of grid to which the i-th row of points is closest.
    In case of ties, it will be snapped to the row of grid with the
    smaller index.
    """
    grid_3d = np.transpose(grid[:, :, np.newaxis], [2, 1, 0])
    diffs = np.sum(np.abs(grid_3d - points[:, :, np.newaxis]), axis=1)
    best = np.argmin(diffs, axis=1)
    return grid[best, :]

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)


# These are the coordinates of th epoints as they appeared in the 3D model
# The order is that of the points given by the pretrained algorithm. This way, the nose
# tip is at: landmarks_ref[face[30]] = landmarks[30]
jaw = [50, 56, 55, 67, 54, 53, 52, 51, 33, 14, 15, 16, 17, 28, 18, 19, 13]
right_eyebrow = [41, 40, 39, 38, 37]  # his right
left_eyebrow = [0, 1, 2, 3, 4]
nose = [34, 36, 35, 29, 43, 42, 30, 5, 6]
right_eye = [60, 59, 58, 57, 62, 61]  # his right
left_eye = [20, 21, 22, 23, 24, 25]
mouth = [47, 45, 44, 31, 7, 8, 10, 11, 12,
         64, 49, 48, 46, 65, 32, 26, 9, 27, 63, 66]
face = [jaw, right_eyebrow, left_eyebrow, nose, right_eye, left_eye, mouth]
face = [y for x in face for y in x] # flattens the array

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	landmarks = predictor(gray, rect)
	landmarks = face_utils.shape_to_np(landmarks)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in landmarks:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	origin = landmarks[31] # Nose tip is index 31
	landmarks = landmarks - origin

	###########
	
	# loads the generic 3D model of the head
	obj_head = ObjLoader()
	obj_head.load_model("Male_Head_landmarks_no_forhead.obj")

	# only take the x and y values for each point
	head_pts = [[vert[0], vert[1]] for vert in obj_head.vert_coords]
	head_pts = np.array(head_pts, np.float32)

	# loads the 3D landmarks of the generic head loaded above
	obj_landmarks = ObjLoader()
	obj_landmarks.load_model("Male_Head_landmarks.obj")

	#TODO could only work with 2 1D array everywhere insteade of 1 2D array

	# only take the x and y values for each point
	landmarks_ref = [[vert[0], vert[1]] for vert in obj_landmarks.vert_coords]
	landmarks_ref = np.array(landmarks_ref, np.float32)

	# center the points on the nose tip
	head_pts = head_pts - landmarks_ref[face[30]]  # Nose tip is index 30
	# unpack the points in 2 arrays, one for x and one for y
	head_pts_x, head_pts_y = zip(*head_pts)
	plt.figure(1)
	plt.scatter(head_pts_x, head_pts_y)

	# center the points on the nose tip
	landmarks_ref = landmarks_ref - landmarks_ref[face[30]]  # Nose tip is index 30
	# unpack the points in 2 arrays, one for x and one for y
	landmarks_ref_x, landmarks_ref_y = zip(*landmarks_ref)
	#plt.figure(1)
	#plt.scatter(landmarks_ref_x, landmarks_ref_y, color='r')

	# The distance between the nose tip and the point at index 0 is considered good by default
	# as to preserve the scale of the landmarks of the 3D model (scale is arbitrarie because of blender). 
	# It's the relative position of all the other points that counts to preserve the shape. 
	# This way, both the scale AND the shape are preserved
	new_model_landmarks = np.array([landmarks_ref[face[0]]])
	factor_ref = landmarks_ref[face[0]]
	factor = landmarks[0]
	print(factor)
	for i in range(1, len(landmarks)):
		if i == 30: # nose tip is at the middle and shouldn't move from there
			new_value = landmarks_ref[face[30]]
		else:
			scale = (landmarks[i])/factor
			new_value = np.array(scale*factor_ref)
		new_model_landmarks = np.append(new_model_landmarks, np.array([tuple(new_value)]), axis=0)

	new_model_landmarks_x, new_model_landmarks_y = zip(*new_model_landmarks)
	
	#fig, ax = plt.subplots()
	#ax.scatter(landmarks_ref_x, landmarks_ref_y, color="g")
	#for i in range(len(landmarks_ref_x)):
	#	ax.annotate(i, (landmarks_ref_x[i], landmarks_ref_y[i]))

	landmarks_ref_x = np.array(landmarks_ref_x)[face]
	landmarks_ref_y = np.array(landmarks_ref_y)[face]

	# Each landmark has a vector of translation from the generic model to the picturels
	translation_x = np.array(new_model_landmarks_x) - np.array(landmarks_ref_x)
	translation_y = np.array(new_model_landmarks_y) - np.array(landmarks_ref_y)

	plt.figure(6)
	start = 0
	end = 18
	print(len(new_model_landmarks_y))

	plt.scatter(new_model_landmarks_x, new_model_landmarks_y, color="g")
	plt.scatter(landmarks_ref_x, landmarks_ref_y, color='r')


	#VECTOR FIELD INTERPOLATION
	# FROM: https://stackoverflow.com/questions/58691789/how-to-interpolate-a-vector-field-with-python

	# To plot vector field (quiver)
	x = landmarks_ref_x
	y = landmarks_ref_y
	u = translation_x
	v = translation_y

	plt.quiver(x, y, u, v)

	xx_line = np.array(np.linspace(-15, 15, 50) * 1000, dtype=np.int)
	yy_line = np.array(np.linspace(-15, 10, 50) * 1000, dtype=np.int)
	xx, yy = np.meshgrid(xx_line, yy_line)

	points = np.array(np.transpose(np.vstack((x, y))) * 1000, dtype=np.int)
	u_interp = interpolate.griddata(points, u * 1000, (xx, yy), method='cubic')
	v_interp = interpolate.griddata(points, v * 1000, (xx, yy), method='cubic')

	plt.figure(3)
	plt.quiver(xx, yy, u_interp, v_interp)

	# Apply vector field (quiver) to head_pts
	# Snap head_pts on the same grid as the vectorfield to apply vectors
	grid = np.column_stack([np.array(xx.flatten()),  np.array(yy.flatten())])

	new_head_pts_x, new_head_pts_y = zip(*head_pts)
	new_head_pts_x = np.array(new_head_pts_x)
	new_head_pts_y = np.array(new_head_pts_y)

	#TODO Use snaped to decide the index of the other array to apply the right vector
	new_head_pts_snaped = ndsnap(head_pts * 1000, grid)
	new_head_pts_snaped = np.array((new_head_pts_snaped), dtype=np.int)
	new_head_pts_snaped_x, new_head_pts_snaped_y = zip(*new_head_pts_snaped)

	###
	indices_x = (new_head_pts_snaped_x - np.min(new_head_pts_snaped_x))
	indices_x = np.array(indices_x * (49 / np.amax(indices_x)), dtype=np.int)
	
	indices_y = (new_head_pts_snaped_y - np.min(new_head_pts_snaped_y))
	indices_y = np.array(indices_y * (49 / np.amax(indices_y)), dtype=np.int)

	k = 0
	for i in range(len(new_head_pts_snaped)):
		u = u_interp[indices_x[k]][indices_y[k]]
		v = v_interp[indices_x[k]][indices_y[k]]
		new_head_pts_snaped[i][0] += u if not math.isnan(u) else 0
		new_head_pts_snaped[i][1] += v if not math.isnan(v) else 0
		k += 1


	new_head_pts_snaped_x, new_head_pts_snaped_y = zip(*new_head_pts_snaped)
	plt.figure(4)
	plt.scatter(new_head_pts_snaped_x, new_head_pts_snaped_y, color="g")
	############

plt.show()
# show the output image with the face detections + facial landmarks
#cv2.imshow("Output", image)
#cv2.waitKey(0)

