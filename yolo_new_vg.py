# version: lstm
# update: 
# lasest date: 22 Dec 2021

# import the necessary packages
from queue import Empty
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import shutil
import imutils
import time
import cv2
import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import ts



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
# 	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-ip", "--ipaddr", required=True,
	help="ip address")

# --------------------------------

ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-th", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")

ap.add_argument("-gt", "--gt", type=bool, default=False,
				help="show ground truth bounding box")
# ap.add_argument("-gtf", "--gtfile", required=False,
# 	            help="path to goundtruth.txt")      
ap.add_argument("-f", "--frame", required=False, type=int,
				help="what frame to be tracked")            
ap.add_argument("-p", "--point", nargs='+', required=False, type=int,
				help="what point coordination will be clicked")  
args = vars(ap.parse_args())



# ------------- Tracking Algo--------------------------

# ใช้ python3.x //3.6
# YOLOv4 ใช้กับ opencv 4.4.0++ 
# ส่วนตัวใช้ opencv 4.5.3 
(major, minor) = cv2.__version__.split(".")[:2]

if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())

else:
	# อัลกอที่เรียกมาสำหรับการทำ Tracking 
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		#"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		#"tld": cv2.TrackerTLD_create,
		#"medianflow": cv2.TrackerMedianFlow_create,
		#"mosse": cv2.TrackerMOSSE_create
	}

	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# -------------End Tracking Algo--------------------------

# ------------- Loading Info from YOLO --------------------
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco-full.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov4.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# ------------- End Loading Info from YOLO -----------------

# open video (input) file
#vs = cv2.VideoCapture(2)
fps = None

# ------------- Init Frames in video --------------------------
writer = None
(W, H) = (None, None)
# # try to determine the total number of frames in the video file
# try:
# 	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
# 		else cv2.CAP_PROP_FRAME_COUNT
# 	total = int(vs.get(prop))
# 	print("[INFO] {} total frames in video".format(total))
# # an error occurred while trying to determine the total
# # number of frames in the video file
# except:
# 	print("[INFO] could not determine # of frames in video")
# 	print("[INFO] no approx. completion time can be provided")
# 	total = -1

# # ------------- End Init Frames in video -----------------------
total = 1

# # ------------- Extract Frames ---------------------------------
# # loop over frames from the video file stream
# count_frame = 1
# while True:
# 	# read the next frame from the file
# 	(grabbed, frame) = vs.read()
# 	# if the frame was not grabbed, then we have reached the end
# 	# of the stream
# 	if not grabbed:
# 		break
# 	# if the frame dimensions are empty, grab them
# 	if W is None or H is None:
# 		(H, W) = frame.shape[:2]
# 	count_frame += 1

# # ------------- End Extract Frames ---------------------------------

# ------------- Set Click Event ------------------------------------
# พอคลิกแล้วมันจะไป promptly update ข้อมูล 
pt = [] 
def click_event(event, x, y, flags, params):
	global pt
	global select
	global nowtracking
	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:
		pt = []
		pt.append([x, y])
		print(pt[0][0], ' ', pt[0][1])
		select = True
		nowtracking = False
		print("clicked!!!")
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_event)

# ------------- End Set Click Event ---------------------------------


# -------------------------------------------------------------------
# init param
track_on_frame = args["frame"] 
track_on_point = args["point"]
locations = [[] for _ in range(total+1)]
prev_frame_time = 0
new_frame_time = 0 
nowtracking = False
select = False
count_frame = 0
fps = None
prev_frame = None
prev_trackClass = None
prev_trackBoxes = None
x0 = 0  # starting point x
y0 = 0  # starting point y
dx = 0  # distance x
dy = 0  # distance y
vxf = 0 # velocity x in frame term
vyf = 0 # velocity y in frame term 
count_firstYolo = 0 
init_frame = 0
frame1 = None
frame2 = None
frame3 = None
frame4 = None
frame5 = None
box2 = None
box3 = None
box4 = None
box5 = None
x_prev = 0
y_prev = 0
# -------------------------------------------------------------------

def detail_object(layerOutputs):
	boxes = []
	confidences = []
	classIDs = []
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			#print("classID", classID)
			confidence = scores[classID]
			if confidence > args["confidence"]:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	return boxes, confidences, classIDs

# ----------------- def detect ------------------------------
def yolo_detect(frame, dx,dy, count_recu):
	#count_recu = 0
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	print("time: ", end-start)
	print("******************** def yolo detect *******************")
	t_boxes, t_confidences, t_classIDs = detail_object(layerOutputs)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	print("found idxs")
	# เลือกเฉพาะตัวที่มี confidence มากสุดในแต่ละคลาส
	idxs = cv2.dnn.NMSBoxes(t_boxes, t_confidences, args["confidence"],
		args["threshold"])
	print("idx----tracking", idxs)
	
	# ----------------- end def detect ------------------------------


	# ----------------- check object------------------------------
	# จะเช็คว่าเป็นวัตถุเดิมกับที่ track อยู่มั้ย

	track_idxs = idxs
	print("check object++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
	# ไล่ดูว่ามีมากกว่า 1 วัตถุที่ถูกจับหรือไม่
	closest = []
	#(x_prev, y_prev, w_prev, h_prev) = [int(v) for v in boxestracker]
	print("len(track_idxs)" ,len(track_idxs))
	nearest = []
	if len(track_idxs) > 0:
			# loop over the indexes we are keeping
			for i in track_idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (t_boxes[i][0], t_boxes[i][1])
				(w, h) = (t_boxes[i][2], t_boxes[i][3])

				print("t_classIDs[i]", t_classIDs[i])
				print("prev_trackClass", prev_trackClass)

				label_class = LABELS[t_classIDs[i]]
				print('label_class', label_class)

				print("color",COLORS[classIDs[i]])
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
				cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				# Check the same class and find the nearest class 

				# Check they re the same class and have distance (prev - curr) less than dist cal from fps
				#if (label_class == prev_trackClass) and (abs(x-x_prev)<= abs(count_nonTrack*vxf)) and (abs(y-y_prev<=abs(count_nonTrack*vyf))):
				#if (label_class == prev_trackClass) and (x-x_prev<= count_nonTrack*vxf) and (abs(y-y_prev<=abs(count_nonTrack*vyf))):
				if (label_class == prev_trackClass):
					print("+++++++++++++++++++++++++++++++++++++++++++++")
					print("same class")
					#print('boxestracker', boxestracker)
					print("x", x)
					print('x_prev', x_prev)
					print("y", y)
					print('y_prev', y_prev)
					print("x-x_prev = ", x-x_prev)
					print("count_nonTrack*vxf =", count_nonTrack*vxf)
					print("y-y_prev = ", y-y_prev)
					print("count_nonTrack*vyf =", count_nonTrack*vyf)

					if dx > 0 and dy > 0: # ++
						print("Vector:  ++ " )
						if x-x_prev < vxf and y-y_prev <vyf:
							print("Vector:    True")
							# เก็บระยะห่างของวัตถุใหม่ที่เจอเทียบกับ prev tracked obj
							# หาระยะที่ห่างจากบ็อกซ์ก่อนหน้า **2 เพราะว่าจะได้ไม่มีปัญหาเรื่องเครื่องหมาย
							# อาจะเกิดปัญหาแบบ -23 < 10 เลยดูเหมือนใกล้ เลยต้องทำให้เครื่องหมายหายไป
							diff = ((x-x_prev) + (y-y_prev))**2

							# เก็บค่่าไว้ใน closest 
							closest.append([i,diff])
					elif dx < 0 and dy > 0: # -+
						print("Vector:  -+ " )
						if x-x_prev > vxf and y-y_prev < vyf:
							print("Vector:  True " )
							diff = ((x-x_prev) + (y-y_prev))**2
							closest.append([i,diff])
					
					elif dx > 0 and dy < 0: # +-
						print("Vector:  +- " )
						if x-x_prev < vxf and y-y_prev > vyf:
							print("Vector:  True " )
							diff = ((x-x_prev) + (y-y_prev))**2
							closest.append([i,diff])

					elif dx < 0 and dy < 0: # --
						print("Vector:  -- " )
						if x-x_prev > vxf and y-y_prev > vyf:
							print("Vector:  True " )
							diff = ((x-x_prev) + (y-y_prev))**2
							closest.append([i,diff])
					
					else:
						continue

					if len(closest)== 0: # วัตถุหาย
						print("Vector:  else case " )
						diff = ((x-x_prev) + (y-y_prev))**2
						closest.append([i,diff])


			if len(closest)== 0: # ไม่มีวัตถุเดียวกัน
				print("no closest")
				#closest.append()
				count_recu += 1
				print("count_recu",count_recu)
				
				if count_recu == 10:
					print("count_recu == 3")
					nowtracking == False
					return None, None, None, None, None
				frame = vid_frame()
				yolo_detect(frame, dx,dy, count_recu)

			# เทียบหาที่ diff น้อยสุด
			if len(closest) != 0:
				nearest = min(closest, key=lambda item:item[1])
		
			return int(nearest[0]), idxs, t_boxes, t_confidences, t_classIDs
	
	if len(track_idxs) == 0:
		print("hi")
		return None, None, None, None, None
	#print("find nearest bbox")

	# return as number ( in order by track_idxs) like 1, 2, 0 
	#return int(nearest[0])

# ----------------- end check object------------------------------


# ----------------- def init tracking ---------------------------
# 

def init_tracking(id_nearest, t_boxes):
	box = (t_boxes[id_nearest][0]-10, t_boxes[id_nearest][1]-10, \
		int(t_boxes[id_nearest][2])+20, int(t_boxes[id_nearest][3]+20))
	print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
	print(box)
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
	tracker.init(frame, box)
	fps = FPS().start()
	#tracker2 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
	#tracker2.init(frame, box)
	nowtracking = True
	TrackClassID = LABELS[t_classIDs[id_nearest]]
	print("TrackClassID",TrackClassID)
#------------------ end init tracking --------------------------- 

#----------------- class lstm ----------------------------------
class LSTMModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(LSTMModel, self).__init__()
		# Hidden dimensions
		self.hidden_dim = hidden_dim

		# Number of hidden layers
		self.layer_dim = layer_dim

		# Building your LSTM
		# batch_first=True causes input/output tensors to be of shape
		# (batch_dim, seq_dim, feature_dim)
		self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

		# Readout layer
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		# Initialize hidden state with zeros
		h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

		# Initialize cell state
		c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

		# 28 time steps
		# We need to detach as we are doing truncated backpropagation through time (BPTT)
		# If we don't, we'll backprop all the way to the start even after going through another batch
		out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

		# Index hidden state of last time step
		# out.size() --> 100, 28, 100
		# out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
		out = self.fc(out[:, -1, :]) 
		# out.size() --> 100, 10
		return out


#----------------- end class lstm ------------------------------

import numpy as np

def save_data():
  [images, labels] = read_data()
  outshape = len(images[0])
  npimages = np.empty((0, outshape), dtype=np.int32)
  nplabels = np.empty((0,), dtype=np.int32)

  for i in range(len(labels)):
	  label = labels[i]
	  npimages = np.append(npimages, [images[i]], axis=0)
	  nplabels = np.append(nplabels, y)

  np.save('images', npimages)
  np.save('labels', nplabels)




def read_data():
  return [np.load('lstm.npy'), np.load('labels.npy')]


def receive_frame():
	frame = None
	data, address = sock.recvfrom(max_length)
		
	if len(data) < 100:
		frame_info = pickle.loads(data)

		if frame_info:
			nums_of_packs = frame_info["packs"]

			for i in range(nums_of_packs):
				data, address = sock.recvfrom(max_length)

				if i == 0:
					buffer = data
				else:
					buffer += data

			frame = np.frombuffer(buffer, dtype=np.uint8)
			frame = frame.reshape(frame.shape[0], 1)

			frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

			if frame is not None and type(frame) == np.ndarray:
				cv2.imshow("Stream", frame)
	if frame is None:
		frame = receive_frame()
	return frame

def vid_frame():
	vs = client.recv()
	# read the next frame from the file
	unique_address, frame = vs
	if frame is None:
		grabbed = False
	#cv2.imshow('frame', frame)
	#cv2.imshow('frame', frame)
	(H, W) = frame.shape[:2]
	frame_dict[unique_address] = frame
	# build a montage using data dictionary
	montages = build_montages(frame_dict.values(), (W, H), (2, 1))
	if frame is None:
		frame = vid_frame()
	
	return frame

#-------- vidgear ------------
from vidgear.gears import NetGear
from imutils import build_montages

# ip addr show
ip_cam = args["ipaddr"]
options = {"multiserver_mode": True}
client = NetGear(
	address= ip_cam,
	port=(5566, 5567),
	protocol="tcp",
	pattern=1,
	receive_mode=True,
	**options
)

# get from uav gps
# x_uav = 13.123456
# y_uav = 100.789123
x_uav = 13.8529355
y_uav = 100.5808003

# init  
height = 6

# # func cal
# Pet = [-2191.365535,-1534.463987, 1758.378694]

# plat0 = x_uav
# plon0 = y_uav
# pe1 = Pet[0]
# pn1 = Pet[1]
# pu1 = Pet[2]


# --------------------main--------------------------------------------
#vs = cv2.VideoCapture('udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264" ! rtph264depay! avdec_h264 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
# vs = client.recv()
# if not vs.isOpened():
#     print('VideoCapture not opened')
# cv2.destroyAllWindows()

import socket
import pickle
import numpy as np

# host = "0.0.0.0"
# port = 5000
# max_length = 90456
# max_length = 1000000

box = None

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind((host, port))
vs = client.recv()
frame_info = None
buffer = None
frame = None

print("-> waiting for connection")

grabbed = True
frame_dict = {}
while True:
	#buffer = None
	#frame = None
	vs = client.recv()
	print("Is_selected", select)
	print("nowtracking", nowtracking)
	# read the next frame from the file
	unique_address, frame = vs
	if frame is None:
		grabbed = False
		frame = vid_frame()
	#cv2.imshow('frame', frame)
	#cv2.imshow('frame', frame)
	(H, W) = frame.shape[:2]
	# frame_dict[unique_address] = frame
	# build a montage using data dictionary
	# montages = build_montages(frame_dict.values(), (W, H), (2, 1))
	print("frame count", count_frame)
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if (frame is not None) and (not grabbed):
		grabbed = True
	elif not grabbed:
		print("last frame")
		break
	
	count_frame += 1

	if count_frame == 1:
		#frame1 = frame
		frame2 = frame
		frame3 = frame
		frame4 = frame
		frame5 = frame
	else:
		frame5 = frame4
		frame4 = frame3
		frame3 = frame2
		frame2 = frame
		#frame1 = frame


	# if count_frame+num_steps-1 > total_frames:
	# 	print("no more next frame available")
	#     break
	print("[INFO] Now at frame : "+ str(count_frame)+" / "+ str(total))
	new_frame_time = time.time()

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if not nowtracking :
		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
		print("time: ", end-start)
		print("********************not tracking*******************")
		boxes, confidences, classIDs = detail_object(layerOutputs)

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		print("found idxs")
		# เลือกเฉพาะตัวที่มี confudence มากสุดในแต่ละคลาส
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])
		print("idx", idxs)

		#---------------------choose tracking frame ------------------#

		# closestBox, box = find_closestBox(frame, idxs, count_frame, select, nowtracking)
		# print("ClosestBox", closestBox)
		# box = (boxes[closestBox[0]][0], boxes[closestBox[0]][1], \
		#     int(boxes[closestBox[0]][2]), int(boxes[closestBox[0]][3]))
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			count_firstYolo += 1
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				
				#locations[count_frame].append([classIDs[i], x + w/2, y + h/2, w, h, confidences[i]]) # tran x to xmid, y to ymid
				
				# !! อย่าลืมแก้ classIDs ให้เป็นคลาสที่เราจะแท็กด้วย !! เพื่อจะโชว์เฉพาะวัตถุที่ต้องการจับ
				# ถ้าไม่ต้องการระบุชัดว่าเอาเฉพาะ คน รถ บลา ๆ ก็ไม่ต้องใส่
				# คน classIds = 0, car -> classIDs = 2, ถ้าอยากดูก็ปริ้นท์มาดูก่อนก็ได้/ ดูใน coco-full.names แล้วไล่ 0 - ... 
				if select == False and nowtracking == False: # and classIDs[i]==2:
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
					cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		#----------------------- LSTM -----------------------------------------------------------
				# crop = frame[y:y+h, x:x+w]
				# data = np.array( crop, dtype='uint8' )
				# np.save( 'lstm' + '.npy', data)
				# label = np.array(([1]), dtype='uint8' )
				# np.save( 'labels' + '.npy', data)


		# ---------------- Check -> at least one frame was detected -----------------------------
				print("track_on_point", track_on_point)
				if count_frame == track_on_frame:
					pt = []
					pt.append(track_on_point)
					#print("track_on_point", track_on_point)
					print(pt[0][0], ' ', pt[0][1])
					select = True
					nowtracking = False

				# check select frame
				if len(idxs) > 0 and select == True and nowtracking == False:
					boxlist = []
					xpt = pt[0][0]
					ypt = pt[0][1]
					for j in idxs.flatten():
						logic = boxes[j][0] < xpt < boxes[j][0] + boxes[j][2] \
							and boxes[j][1] < ypt < boxes[j][1] + boxes[j][3]
						# choose the bounding box from how close to the center  
						# of click coordinate
						if logic == True:
							centerXbox = (boxes[j][0] + boxes[j][2]) / 2
							centerYbox = (boxes[j][1] + boxes[j][3]) / 2
							a = np.array([centerXbox,centerYbox])
							b = np.array([xpt,ypt])
							dist = np.linalg.norm(a-b)
							boxlist.append([j,dist])
					
					if len(boxlist) == 0 :
						continue
					
					# เลือกอันที่ใกล้ที่สุด หา min ใน boxlist โดยดูที่ค่า item[1]==dist
					closestBox = min(boxlist, key=lambda item:item[1])
					print("ClosestBox", closestBox)
					box = (boxes[closestBox[0]][0]-10, boxes[closestBox[0]][1]-10, \
						int(boxes[closestBox[0]][2])+20, int(boxes[closestBox[0]][3]+20))
					print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
					print(box)
					tracker1 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
					tracker1.init(frame, box)
					print("frame size",frame.shape)

					# - lstm
					print('0',box[0], '1', box[1], '2', box[2], '3', box[3])
					if box[1]>= box[0] and box[2]>=box[3]:
						crop = frame[box[0]:box[1], box[3]:box[2]] #ต้องได้เป็น diffx diff y แต่ไม่แน่ใจว่า 0 1 2 3 นี่คือเรียงไง
					
					elif box[0]>=box[1] and box[2]>=box[3]:
						crop = frame[box[1]:box[0], box[3]:box[2]]

					elif box[1]>=box[0] and box[3]>=box[2]:
						crop = frame[box[0]:box[1], box[2]:box[3]]

					elif box[0]>=box[1] and box[3]>=box[2]:
						crop = frame[box[1]:box[0], box[2]:box[3]]
					print('crop',crop)
					data = np.array([crop])
					#data = torch.from_numpy(crop)
					#data /= 255.0  
					print("data", data)
					#np.save( 'lstm' + '.npy', data)
					label_lstm = np.array(([1]), dtype='uint8' )
					#np.save( 'labels' + '.npy', data)
					#-- lstm --
					#save_data() # ได้
					batch_size = 100
					# data = torch.utils.data.DataLoader(dataset=data, 
					#                        batch_size=batch_size, 
					#                        shuffle=True)

					lstm_start = time.time()
					print("LSTM")
					input_dim = 28
					hidden_dim = 100
					layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
					output_dim = 10
					num_epochs = 5
					seq_dim = 28
					learning_rate =0.1
					criterion = nn.CrossEntropyLoss()

					model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
					optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

					data = torch.Tensor(data)
					
					
					label_lstm = torch.Tensor(label_lstm)
					label_lstm = label_lstm.type(torch.LongTensor)
					print('image_test', data.shape)

					iter = 0

					for epoch in range(num_epochs):
						print("LSTM for epoch loop")
						for i, images in enumerate(data):

							images.resize_(1,1, 28, 28)
							# Load images as torch tensor with gradient accumulation abilities
							images = images.view(-1, seq_dim, input_dim).requires_grad_()
							#print('image view', images[0])
							print("lstm model")

							# Clear gradients w.r.t. parameters
							optimizer.zero_grad()

							# Forward pass to get output/logits
							# outputs.size() --> 100, 10
							outputs = model(images)
							

							# Calculate Loss: softmax --> cross entropy loss
							loss = criterion(outputs, label_lstm)
							

							# Getting gradients w.r.t. parameters
							loss.backward()

							# Updating parameters
							optimizer.step()

							iter += 1

							if iter % 1 == 0:
								# Calculate Accuracy         
								correct = 0
								total = 0
								# Iterate through test dataset
								for images in data:
									images.resize_(1,1, 28, 28)
									# Resize image
									images = images.view(-1, seq_dim, input_dim)

									# Forward pass only to get logits/output
									outputs = model(images)

									# Get predictions from the maximum value
									_, predicted = torch.max(outputs.data, 1)

									# Total number of labels
									total += label_lstm.size(0)

									# Total correct predictions
									correct += (predicted == label_lstm).sum()

								accuracy = 100 * correct / total

								# Print Loss
								print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


					fps = FPS().start()
					#tracker2 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
					#tracker2.init(frame, box)
					nowtracking = True
					TrackClassID = LABELS[classIDs[closestBox[0]]]
					print("TrackClassID",TrackClassID)
					lstm_stop = time.time()
					print("lstm time =",lstm_stop-lstm_stop)

				#elif len(idxs) == 0 and select == True and nowtracking == False:
				if box is None:
					select = False

	if box is None:
		select = False
	# ครั้งแรกที่ yolo เจอวัตถุแต่ยังไม่ได้เลือก
	if (count_firstYolo==1):
		frame2 = frame
		frame3 = frame
		frame4 = frame
		frame5 = frame
		if select :
			box2 = box
			box3 = box
			box4 = box
			box5 = box
		x0 = x 
		y0 = y 
		init_frame = count_frame
		count_prev_frame = count_frame
	
	# ไม่ใช่ครั้งแรกที่ yolo เจอ และ user เลือกวัตถุแล้ว
	elif (count_firstYolo>1 and select):
		if box5 == None:
			box2 = box
			box3 = box
			box4 = box
			box5 = box
		
		if (count_frame-init_frame < 5):
			dx = x-x0
			dy = y-y0
			# จะได้เป็น 1 เฟรมขยับเท่าไหร่
			vxf = dx/(count_frame-init_frame)
			vyf = dy/(count_frame-init_frame)
		else:
			dx = box[0] - box5[0]
			dy = box[1] - box5[1]
			vxf = dx/5
			vyf = dy/5
		
		# shift ค่า
		box5 = box4
		box4 = box3
		box3 = box2
		box2 = box
		print("count_frame", count_frame)
		print("count_prev_frame", count_prev_frame)
		print("init_frame", init_frame)
		print("dx ", dx)
		print("dy ", dy)
		print("vxf ", vxf)
		print("vyf ", vyf)
		print("count_frame - init_frame", count_frame - init_frame)

	elif (count_firstYolo>1 and not select):

		# dx = x-x0
		# dy = y-y0
		print("dx++++++++++++++++", dx , "+++++++++++++++++  first")
		# vxf = dx/(count_frame - init_frame)
		# vyf = dy/(count_frame - init_frame)
		# #vxf = dx/(count_frame - prev_frame)
		# #vyf = dy/(count_frame - prev_frame)
		
		# print("vxf ", vxf)
		# print("vyf ", vyf)
		# print("count_frame - init_frame", count_frame - init_frame)

	

	if nowtracking:
		print("-----------------nowtracking--------------------")
		(success, boxestracker) = tracker1.update(frame)

		print('boxestracker_nowtracking', boxestracker)

		# ----------------------if not success----------------------------------
		# ตั้งใจว่าจะให้ไปเรียก yolo มาdetect แล้วหาตำแหน่งที่ใกล้ที่สุดทั้ง 4 มุมกรอบ + ประเภทเดียวกัน
		# เลือกกรอบนั้น + ขยายสัก ด้านละ 10 
		# ต้องดูว่ามี confident >  50 ด้วย
		if not success:
			count_nonTrack += 1
			print("lose Tracking")
			#prev_frame = frame
			count_recu = 0
			try:
				id_nearest, track_idxs, t_boxes, t_confidences, t_classIDs= yolo_detect(frame, dx, dy,count_recu)
			
			except:
				id_nearest = None
				nowtracking == False

			if id_nearest == None:
				continue

			if id_nearest != None:
				#init_tracking(id_nearest, t_boxes)
				box = (t_boxes[id_nearest][0], t_boxes[id_nearest][1], \
					int(t_boxes[id_nearest][2]), int(t_boxes[id_nearest][3]))
				print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
				print(box)
				#tracker2 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
				#tracker2.init(frame, box)
				fps = FPS().start()
				#tracker2 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
				#tracker2.init(frame, box)
				nowtracking = True
				TrackClassID = LABELS[t_classIDs[id_nearest]]
				#TrackClassID = "Track2"

				(success, boxestracker) = tracker1.update(frame)
				print("success - id_nreaest", success)
			else:
				continue


		# วาดกรอบตัวที่เลือกมา
		if success:
			print("success Tracking")
			count_nonTrack = 0
			(x, y, w, h) = [int(v) for v in boxestracker]
			x_prev = x
			y_prev = y
			prev_frame = frame
			count_prev_frame = count_frame
			prev_trackBoxes = boxestracker
			prev_trackClass = TrackClassID
			trackcolor = (0, 255, 0)
			cv2.rectangle(frame, (x, y), (x + w, y + h),trackcolor, 2)
			text = "{}".format(str(TrackClassID))
			cv2.putText(frame, text, (x, y ),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, trackcolor, 2)
		# update the FPS counter
		fps.update()
		fps.stop()
		# initialize the set of information we'll be displaying on
		# the frame

		# # ----------------------if not success----------------------------------
		# # ตั้งใจว่าจะให้ไปเรียก yolo มาdetect แล้วหาตำแหน่งที่ใกล้ที่สุดทั้ง 4 มุมกรอบ + ประเภทเดียวกัน
		# # เลือกกรอบนั้น + ขยายสัก ด้านละ 10 
		# # ต้องดูว่ามี confident >  50 ด้วย
		# if not success:
		#     print("lose tracking")
		#     #prev_frame = frame
		#     track_idxs, t_boxes, t_confidences, t_classIDs= yolo_detect(frame)
		#     id_nearest = check_object(track_idxs, t_boxes, t_classIDs)
		#     init_tracking(id_nearest, t_boxes)



		#-----------------------------------------------------------------------

		# Geolocation 
		# func cal
		Pet = [-2191.365535,-1534.463987, 1758.378694]

		# มาจากตำแหน่งของวัตถุในรูป (x,y,w,h) * ขนาดเฟรม? // หาขนาดเฟรมโดยใช้โคด
		pu = x*1024
		pv = y*576
		# pe1 = Pet[0]
		# pn1 = Pet[1]
		# pu1 = Pet[2]

		# ob_lat = ts.get_lat(plat0, plon0, pe1, pn1, pu1)
		# ob_lon = ts.get_lon(plat0, plon0, pe1, pn1, pu1)

		box_lat = ts.findlat(pu, pv)
		box_lon = ts.findlon(pu, pv)


		print("lat: ", box_lat)
		print("lon: ", box_lon)

		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
			("lat", box_lat),
			("lon", box_lon),
		]
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) == ord('q'):
		break
	
	# check if the video writer is None
	print("writer",writer)
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))


	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.stop()


