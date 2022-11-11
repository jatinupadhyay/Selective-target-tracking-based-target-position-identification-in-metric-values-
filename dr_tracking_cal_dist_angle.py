from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import math
import numpy as np
import time
import os
import logging
import xlsxwriter
from math import cos,radians,sin
from xlsxwriter.utility import xl_rowcol_to_cell
from collections import deque
workbook = xlsxwriter.Workbook('obj_loc.xlsx')
worksheet = workbook.add_worksheet() 
new_frame_time = 0
prev_frame_time = 0
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def distance(x1,y1,x2,y2):
#dis = dist.euclidean((x1,y1), (x2,y2))
	return math.sqrt((math.fabs(x2-x1))**2+((math.fabs(y2-y1)))**2)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())
ap.add_argument("-b", "--buffer", type=int, default=10000,
	help="max buffer size")
args = vars(ap.parse_args())

# trackers = cv2.MultiTracker_create()
pts = deque(maxlen=args["buffer"])

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]


if int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())


else:
	
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,#accurate but slower then kcf and mosse 
		"kcf": cv2.TrackerKCF_create,# fast but inaccurate
		"boosting": cv2.legacy_TrackerBoosting.create,# faster then csrt, but not accurate if target is overlapped  
		"mil": cv2.TrackerMIL_create,# slow, sensitive to the object movement, quit in accurate compare to otheres 
		"tld": cv2.legacy_TrackerTLD.create,#error  and sensitive 
		"medianflow": cv2.legacy_TrackerMedianFlow.create,#inaccurate 
		"mosse": cv2.legacy_TrackerMOSSE.create# fast but inaccurate for small size object 
	}

	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=1000)
	(H, W) = frame.shape[:2]
	hech=frame.shape[0]
	wu=frame.shape[1]
# 	fps.update()
# 	fps.stop()
# 	info = [
#  			("Tracker", args["tracker"]),
# #  			("Success", "Yes" if success else "No"),
#  			("FPS", "{:.2f}".format(fps.fps())),
#  			]

	# check to see if we are currently tracking an object
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		(x, y, w, h) = [int(v) for v in box]
# 		cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
# 		(centerX, centerY, width, height) = box.astype("int")
		(cX,cY)=(midpoint((x, y), (x + w, y + h)))
		dX=int(cX)
		dY=int(cY)
		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
# 			(centerX, centerY, width, height) = box.astype("int")
			(cX,cY)=(midpoint((x, y), (x + w, y + h)))
			dX=int(cX)
			dY=int(cY)
			
		# update the FPS counter

# 		for box in box:
# 			(x, y, w, h) = [float(v) for v in box]
# 		print (int(v) for v in box)
# 			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 			(centerX, centerY, width, height) = box.astype("float")
# 			(cX,cY)=(midpoint((x, y), (x + w, y + h)))
		
		
			dX=int(cX)
			dY=int(cY)
			(dirX,dirY)=("","")
		#print(centerX,centerY)
# 		fps.update()
# 		fps.stop()
# 
# # 		initialize the set of information we'll be displaying on the frame
# 		info = [
#  			("Tracker", args["tracker"]),
# #  			("Success", "Yes" if success else "No"),
# #  			("FPS", "{:.2f}".format(fps.fps())),
#  		]
		####################################################################
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		
			center_l = (dX,dY)
 	
			pts.appendleft(center_l)
		for i in range(1, len(pts)):
			if pts[i - 1] is None or pts[i] is None:
				continue
 
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
			#thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)
		####################################################################		
			cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),(10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,0.60, (255, 0, 0), 1)
		#print(dX,dY)
		#a.append([dX,dY])
		#cen.append([centerX,centerY])
		#print(a)
		#print(cen)
		#cen=[]
		#a=[]
		x1=int(wu/2)
		y1=int(hech/2)
		D = distance(x1,y1, dX,dY)
		DIS_OBJ= D/90
		#len.append([DIS_OBJ])
		#len=[]
		#print(DIS_OBJ)
		(mX, mY) = midpoint((x1,y1), (dX,dY))
		hypotenuse=distance(x1,y1,dX,dY)
		adjecent=distance(dX,y1,x1,y1)+0.00000000001
		opposite=distance(dX,y1,dX,dY)+0.00000000001
		cv2.line(frame,(0,y1),(wu,y1),(0,0,255,100),4)
		angle = int(math.atan((dY)/(dX+0.00000000000000000000000000001))*360/math.pi)
		if dX>=(wu/2) and dY<=(hech/2):
			#print ("First quadrent")
			cv2.putText(frame, "1st", (int(mX+10), int(mY + 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 1)
			agl=(np.arctan((opposite)/(adjecent)) * 180/math.pi)
			#anglo.append(agl)
			print(x1,",",y1,",",dX,",",dY,",",agl,",",DIS_OBJ,",",D)
			#anglo=[]
		elif dX>(wu/2) and dY>(hech/2):
			#print ("Fourth Quadrent")
			cv2.putText(frame, "4th", (int(mX+30), int(mY + 30)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 1)
			agl=(np.arctan((adjecent )/(opposite)) * 180/math.pi)+270
			#anglo.append(agl)
			print(x1,",",y1,",",dX,",",dY,",",agl,",",DIS_OBJ,",",D)
			#anglo=[]
		elif dX<=(wu/2) and dY<=(hech/2):
			#print ("second quadrent")
			cv2.putText(frame, "2nd", (int(mX+60), int(mY + 60)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 1)
			agl=(np.arctan((adjecent )/(opposite)) * 180/math.pi)+90
			#anglo.append(x1,y1,dX,dY,agl,DIS_OBJ)
			print(x1,",",y1,",",dX,",",dY,",",agl,",",DIS_OBJ,",",D)
			#anglo=[]
		elif dX<(wu/2) and dY>(hech/2):
			#print ("third Quadrent")
			cv2.putText(frame, "3rd", (int(mX+100), int(mY + 100)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 1)
			agl=(np.arctan((opposite)/(adjecent)) * 180/math.pi)+180
			#anglo.append(agl)
			print(x1,",",y1,",",dX,",",dY,",",agl,",",DIS_OBJ,",",D)
			#anglo=[]
		else:
			agl=0
			print("null")
			
#################################################################################################################################
		if agl>0 and agl<90:
			new_agl= agl+180
		elif agl>90 and agl<180:
			new_agl=360-(180-agl)
		elif agl>180 and agl<270:
			new_agl=180-agl
		else:
			new_agl=180-(360-agl)
		cosine=100.58*cos(radians(344.45))
		sine=100.58*sin(radians(344.45))
		
		x_1d= (dX)+int(cosine)
		y_1d= -((-dY)+int(sine))
		#D1 = distance(x1,y1, x_2,y_2)
		#DIS_OBJ_1= D1/47
		#print(DIS_OBJ_1)
		#print(x_2,y_2)
		#cv2.line(frame, (x1,y1), (x_2,y_2),(0,0,0), 2)					
#################################################################################################################################
		
		edges=(dX,dY)
		cosine=(1.94*24)*cos(radians(267.69))
		sine=(1.94*24)*sin(radians(267.69))
		
		x_2= (dX)+int(cosine)
		y_2= -((-dY)+int(sine))
		D1 = distance(x1,y1, x_2,y_2)
		DIS_OBJ_1= D1/47
		print(DIS_OBJ_1)
		print(x_2,y_2)
		#cv2.line(frame, (x1,y1), (x_2,y_2),(0,0,0), 2)
###################################################################################################################################
		cosine3=47.04*cos(radians(218.11))##Angle=41.347777219687984 ,Dist= 1.417091557901848
		sine3=47.04*sin(radians(218.11))#38.92754359278692 , 0.6891041223533362 , 66.84309986827361
		
		x_3= (dX)+int(cosine3)
		y_3= -((-dY)+int(sine3))
		D3 = distance(x1,y1, x_3,y_3)
		DIS_OBJ_3= D3/47
		
		D4 = distance(x_2,y_2, x_3,y_3)
		DIS_OBJ_4= D4/47
		print(DIS_OBJ_3)
		print(DIS_OBJ_4)
		print(x_3,y_3) 
		#cv2.line(frame, (x1,y1), (x_3,y_3),(0,0,0), 2)
		#cv2.line(frame, (x_2,y_2), (x_3,y_3),(0,0,0), 2)
########################################################################################################################################
		h3_desh=(202)
		k3_desh=(114)
		h_desh=(202-x_3)
		k_desh=(114-y_3)
		print(h_desh,k_desh)
		print("##################")
		print(x1,y1)
		print("##################")
		x1_desh=x_1d+h_desh
		y1_desh=y_1d+k_desh
		
		x2_desh=x_2+h_desh
		y2_desh=y_2+k_desh
		print("##################")
		print(x1_desh,y1_desh)
		print(x2_desh,y2_desh)
		print(h_desh,k_desh)
		print("##################")
		# cv2.circle(frame,(h3_desh,k3_desh),5, (0, 0, 0), -1 )
# 		cv2.circle(frame,(x1_desh,y1_desh),5, (0, 0, 0), -1 )
# 		cv2.circle(frame,(x2_desh,y2_desh),5, (0, 0, 0), -1 )
# 		d4_desh= distance(x1_desh,y1_desh, h3_desh,k3_desh)
# 		print("d4_desh=",d4_desh)
# 		print("D3:",D3)
# 		cv2.line(frame, (x1_desh,y1_desh), (x2_desh,y2_desh),(0,0,0), 2)
# 		cv2.line(frame, (x1_desh,y1_desh), (h3_desh,k3_desh),(0,0,0), 2)
# 		cv2.line(frame, (x2_desh,y2_desh), (h3_desh,k3_desh),(0,0,0), 2)
		# del_x=202-int(cosine3)
# 		(del_y)=114-int(sine3)
# 		print(del_x,del_y)
# 		agl_n1=(np.arctan((del_x-x1)/(del_y-x2)) * 180/math.pi))
# 		agl_n2=(np.arctan((del_x-x_1)/(del_y-y_2)) * 180/math.pi)
# 		cv2.circle(frame,(del_x,del_y),5, (0, 0, 0), -1 )
# 		cosine4=D3*cos(radians(agl_n1))
# 		sine4=D3*sin(radians(agl_n1))
# 		x1_new= (dX)+int(cosine3)
# 		y1_new= -((-dY)+int(sine3))
# 		print(x1_new,y1_new)
# 		cv2.circle(frame,(x1_new,y1_new),5, (0, 0, 0), -1 )
		
########################################################################################################################################
 ### For moved object from image 3 h3_desh,k3_desh
########################################################################################################################################		
		dX_d3= distance(dX,dY, h3_desh,k3_desh)
		hypotenuse3=distance(h3_desh,k3_desh,dX,dY)
		adjecent3=distance(dX,k3_desh,h3_desh,k3_desh)+0.00000000001
		opposite3=distance(dX,k3_desh,dX,dY)+0.00000000001
		#cv2.line(frame,(0,k3_desh),(wu,k3_desh),(0,0,255,100),4)
		dis4= distance(dX,dY, h3_desh,k3_desh)
		print(h3_desh,k3_desh)
		if h3_desh>=(dX) and k3_desh<=(dY):
			print ("1 quadrent")
			agl4=(np.arctan((opposite3)/(adjecent3)) * 180/math.pi)
			#anglo.append(agl4)
			print(agl4,",",dX_d3)
			#anglo=[]
		elif h3_desh>(dX) and k3_desh>(dY):
			print ("4 Quadrent")
			agl4=(np.arctan((adjecent3 )/(opposite3)) * 180/math.pi)+270
			#anglo.append(agl4)
			print(agl4,",",dX_d3)
			#anglo=[]
		elif h3_desh<=(dX) and k3_desh<=(dY):
			print ("2 quadrent")
			agl4=(np.arctan((adjecent3 )/(opposite3)) * 180/math.pi)+90
			#anglo.append(h3_desh,k3_desh,dX,dY,agl4,dis4)
			print(agl4,",",dX_d3)
			#anglo=[]
		elif h3_desh<(dX) and k3_desh>(dY):
			print ("3 Quadrent")
			agl4=(np.arctan((opposite3)/(adjecent3)) * 180/math.pi)+180
			#anglo.append(agl4)
			print(agl4,",",dX_d3)
			#anglo=[]
		else:
			agl4=0
			print("null")
			
		cosine_h3=dX_d3*cos(radians(agl4))
		sine_h3=dX_d3*sin(radians(agl4))
		
		x_3_h3= (dX)+int(cosine3)
		y_3_k3= -((-dY)+int(sine3))
		print("x_3_h3,y_3_k3:",x_3_h3,y_3_k3)
########################################################################################################################################
########################################################################################################################################
 ### For moved object from image 2 x2_desh,y2_desh
########################################################################################################################################		
		dX_d2= distance(dX,dY, x2_desh,y2_desh)
		hypotenuse2=distance(x2_desh,y2_desh,dX,dY)
		adjecent2=distance(dX,y2_desh,x2_desh,y2_desh)+0.00000000001
		opposite2=distance(dX,y2_desh,dX,dY)+0.00000000001
		#cv2.line(frame,(0,y2_desh),(wu,y2_desh),(0,0,255,100),4)
		dis5= distance(dX,dY, x2_desh,y2_desh)
		print(x2_desh,y2_desh)
		if x2_desh>=(dX) and y2_desh<=(dY):
			print ("1 quadrent")
			agl5=(np.arctan((opposite2)/(adjecent2)) * 180/math.pi)
			#anglo.append(agl5)
			print("agl5,",agl5,",",dX_d2)
			#anglo=[]
		elif x2_desh>(dX) and y2_desh>(dY):
			print ("4 Quadrent")
			agl5=(np.arctan((adjecent2 )/(opposite2)) * 180/math.pi)+270
			#anglo.append(agl5)
			print("agl5,",agl5,",",dX_d2)
			#anglo=[]
		elif x2_desh<=(dX) and y2_desh<=(dY):
			print ("2 quadrent")
			agl5=(np.arctan((adjecent2 )/(opposite2)) * 180/math.pi)+90
			#anglo.append(x2_desh,y2_desh,dX,dY,agl5,dis5)
			print("agl5,",agl5,",",dX_d2)
			#anglo=[]
		elif x2_desh<(dX) and y2_desh>(dY):
			print ("3 Quadrent")
			agl5=(np.arctan((opposite2)/(adjecent2)) * 180/math.pi)+180
			#anglo.append(agl5)
			print("agl5,",agl5,",",dX_d2)
			#anglo=[]
		else:
			agl5=0
			print("null")
########################################################################################################################################
########################################################################################################################################
 ### For moved object from image 1 x1_desh,y1_desh
########################################################################################################################################		
		dX_d1= distance(dX,dY, x1_desh,y1_desh)
		hypotenuse1=distance(x1_desh,y1_desh,dX,dY)
		adjecent1=distance(dX,y1_desh,x1_desh,y1_desh)+0.00000000001
		opposite1=distance(dX,y1_desh,dX,dY)+0.00000000001
		#cv2.line(frame,(0,y1_desh),(wu,y1_desh),(0,0,255,100),5)
		dis5= distance(dX,dY, x1_desh,y1_desh)
		print(x1_desh,y1_desh)
		if x1_desh>=(dX) and y1_desh<=(dY):
			print ("1 quadrent")
			agl6=(np.arctan((opposite1)/(adjecent1)) * 180/math.pi)
			#anglo.append(agl6)
			print(agl6,",",dX_d1)
			#anglo=[]
		elif x1_desh>(dX) and y1_desh>(dY):
			print ("4 Quadrent")
			agl6=(np.arctan((adjecent1 )/(opposite1)) * 180/math.pi)+270
			#anglo.append(agl6)
			print(agl6,",",dX_d1)
			#anglo=[]
		elif x1_desh<=(dX) and y1_desh<=(dY):
			print ("2 quadrent")
			agl6=(np.arctan((adjecent1 )/(opposite1)) * 180/math.pi)+90
			#anglo.append(x1_desh,y1_desh,dX,dY,agl6,dis5)
			print(agl6,",",dX_d1)
			#anglo=[]
		elif x1_desh<(dX) and y1_desh>(dY):
			print ("3 Quadrent")
			agl6=(np.arctan((opposite1)/(adjecent1)) * 180/math.pi)+180
			#anglo.append(agl6)
			print(agl6,",",dX_d1)
			#anglo=[]
		else:
			agl6=0
			print("null")
########################################################################################################################################
		
# 		cv2.setMouseCallback("frame", click_and_crop)
#  
# # keep looping until the 'q' key is pressed
# 		while True:
# 	# display the image and wait for a keypress
# 			cv2.imshow("frame", frame)
# 			key = cv2.waitKey(1) & 0xFF
#  
# 	# if the 'r' key is pressed, reset the cropping region
# 			if key == ord("r"):
# 				image = clone.copy()
#  
# 	# if the 'c' key is pressed, break from the loop
# 			elif key == ord("c"):
# 				break
#  
# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# 		if len(refPt) == 2:
# 			roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
# 			cv2.imwrite("ROI", roi)
# 			cv2.waitKey(0)
########################################################################################################################################
		# fps.update()
# 		fps.stop()
		fps.update()
		fps.stop()

# 		initialize the set of information we'll be displaying on the frame
		info = [
 			("Tracker", args["tracker"]),
#  			("Success", "Yes" if success else "No"),
#  			("FPS", "{:.2f}".format(fps.fps())),
 		]
	# 	print("h3_desh,k3_desh",h3_desh,k3_desh,"agl4",agl4,"dX_d1",dX_d3)
# 		print("x2_desh,y2_desh",x2_desh,y2_desh,"agl5",agl5,"dX_d1",dX_d2)
# 		print("x1_desh,y1_desh",x1_desh,y1_desh,"agl6",agl6,"dX_d1",dX_d1)  
		cv2.putText(frame, "{:.2f}M".format(DIS_OBJ), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)
# 		cv2.putText(frame, "{:.2f}deg".format(agl), (int(x1), int(y1 + 2)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)
		cv2.line(frame, (x1,y1), (dX,dY),(0,255,0), 2)
		
		cv2.line(frame,(x1,0),(x1,hech),(0,0,255,100),4)
		#cv2.line(frame,(dX,dY),(x_2,y_2),(0,100,255,100),4)
		#cv2.line(frame,(dX,dY),(x_2,y_2),(0,100,255,100),4)
		#cv2.line(frame,(x1, y1),(x_2,y_2),(10,100,255,100),4)
		#cv2.line(frame,(dX,dY),(x_3,y_3),(0,100,255,100),4)
		#cv2.line(frame,(dX,dY),(x_1d,y_1d),(0,100,255,100),4)
		#cv2.line(frame,(x1, y1),(x_3,y_3),(10,100,255,100),4)
		#cv2.line(frame,(x1, y1),(273,203),(0,100,255,100),4)
		#cv2.line(frame,(dX,dY),(273,203),(0,100,255,100),4)
		# cv2.circle(frame, (x_2,y_2), 5, (0, 150, 100), -1)
# 		cv2.circle(frame, (x_3,y_3), 5, (0, 150, 100), -1)
		cv2.circle(frame, (dX, dY), 5, (0, 255, 0), -1)
		cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
		# cv2.circle(frame, (x_1d, y_1d), 5, (0, 255, 0), -1)
# 		cv2.circle(frame, (x_2, y_2), 5, (0, 255, 150), -1)
# 		cv2.circle(frame, (dX, y1), 5, (0, 255, 0), -1)
		
		##########################################################
# 		cv2.line(frame, (x_1d,y_1d), (x_2,y_2),(0,0,0), 2)
# 		cv2.line(frame, (x_1d,y_1d), (x_3,y_3),(0,0,0), 2)
# 		cv2.line(frame, (x_2,y_2), (x_3,y_3),(0,0,0), 2)
		##########################################################
		#cv2.HoughLines(edges,1, 76, 272, 200);
		#cv2.circle(frame,(dX, dY), 76, (0,0,255), -1)
		#cv2.selectROI("Frame", (dX, dY), fromCenter=False, showCrosshair=True)
		# worksheet.write('A1', "x1")
# 		worksheet.write('B1', "y1")
# 		worksheet.write('C1', "dX")
# 		worksheet.write('D1', "dY")
# 		worksheet.write('E1', "agl")
# 		worksheet.write('F1',"DIS_OBJ")
# 		#worksheet.write('G1',"new_agl")
# 		worksheet.write('A2', x1)
# 		worksheet.write('B2', y1)
# 		worksheet.write('C2', dX)
# 		worksheet.write('D2', dY)
# 		worksheet.write('E2', agl)
# 		worksheet.write('F2',DIS_OBJ)
# 		#worksheet.write('G2',new_agl)
		workbook.close()
		cv2.putText(frame, "agl:{}".format(agl),(100,100), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 100, 0), 2)
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (100,50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# 			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		tracker.init(frame, initBB)
		fps = FPS().start()

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()