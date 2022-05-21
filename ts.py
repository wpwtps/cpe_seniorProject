from __future__ import print_function

from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
import time
import math
from pymavlink import mavutil
import pymap3d
from math import radians, cos, sin, asin, sqrt, atan2
import argparse  
import numpy as np

height = -9

f = 0.00367 #focal length
# f = 3.67
cx = 1024/2 #width/2
cy = 576/2 #height/2
fx = 595.52493169
fy = 597.25551576
def changetorad(deg):
    return (deg  * math.pi/180)
    # return radians(deg)

angle = changetorad(0)
# angle = changetorad(-10)
# print(radians(45),radians(90), radians(-45))
def lat_uav():
    # return vehicle.location.global_frame.lat
    return 13.847157 # location at incee
    # return 13.846064 #location com de. inside
    # return 13.8402563 # location office
    # return 13.8462743 # location com de. outside
    # return 13.840793 # test
    # return 13.847150 # car
    # return 13.847289048697506 # car change

def lon_uav():
    # return vehicle.location.global_frame.lon
    return 100.565288 # location at incee
    # return 100.5685915 # location com de. inside
    # return 100.5607318 # location office
    # return 100.5686319 # location com de.
    # return 100.561119
    # return 100.565440 #car
    # return 100.56515604502513 #car change

def get_pitch():
    # if type(math.degrees(vehicle.attitude.pitch)) == 'float':
    #     return math.degrees(vehicle.attitude.pitch)
    # else:
    #     return 0
    # print(changetorad(27.71))
    # return changetorad(4.22)
    # return changetorad(-75.9)
    return changetorad(-20)
    # return changetorad(-1.38) #+5
    # return changetorad(-20)
    # return 4.22
    # return changetorad(45)
    # return changetorad(39.14)
    # return changetorad(14.1+15)
    # return changetorad(30.21)
    # return changetorad(-30.48)
    # return changetorad(43.84)
    # return changetorad(0)
    # return radians(0.07)

def get_roll():
    # if type(math.degrees(vehicle.attitude.roll)) == 'float':
    #     return math.degrees(vehicle.attitude.roll)
    # else:
        # return 0
    # return changetorad(0.38)
    # return changetorad(-3.74)
    return changetorad(3)
    # return changetorad(0.25)
    # return changetorad(10.56)
    # return changetorad(15)
    # return changetorad(11.50)
    # return changetorad(0.52)
    # return changetorad(16.77)
    # return changetorad(0)
    # return radians(0.52)
    
def get_yaw():
    # if type(math.degrees(vehicle.attitude.yaw)) == 'float':
    #     return math.degrees(vehicle.attitude.yaw)
    # else:
    #     return 0
    # return changetorad(15.57)
    # return changetorad(25.91)
    # return changetorad(21.76)
    # return changetorad(303.25)
    # return changetorad(144.33)
    # return changetorad(100)
    # return changetorad(303.84+19)
    # return changetorad(155)
    # return changetorad(104.42)
    # return changetorad(102+90)
    # return changetorad(82.84+90)
    # return changetorad(0)
    return changetorad(155)
    # return changetorad(0)
# def test_get_lat(lat0, lon0, h0, e1, n1, u1):
#     rx = np.array([[1, 0, 0], 
#                 [0, cos(roll), sin(roll)],
#                 [0, -sin(roll), cos(roll)]])

def get_lat(lat0, lon0, h0, e1, n1, u1):
    # print(lat0, lon0, h0, e1, n1, u1)
    # if((e1 * n1) > 0):
    #     e1 = e1 * -1
    #     n1 = n1 * -1
    lat1, lon1, h1 = pymap3d.enu2geodetic(e1, n1, u1, lat0, lon0, h0, ell=None, deg=True)
    return lat1

def get_lon(lat0, lon0, h0, e1, n1, u1):
    # if((e1 * n1) > 0):
    #     e1 = e1 * -1
    #     n1 = n1 * -1
    # e1 = 80,n1 = 30
    lat1, lon1, h1 = pymap3d.enu2geodetic(e1, n1, u1, lat0, lon0, h0, ell=None, deg=True)
    return lon1

def Distance(lat1, lon1, lat2, lon2): 
      
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371
    # print("Distance ", c*r)
    # calculate the result 
    return(c * r * 1000) 

def velocity(lat0, lon0, lat1, lon1, fps):
    # lat0 = round(lat0, 6)
    # lon0 = round(lon0, 6)
    print("Python is ",lat0, lon0, lat1, lon1)
    fps = 15
    distance = Distance(lat0, lon0, lat1, lon1)
    # print("Distance ",distance)
    time = 2/fps
    v = (distance/time) *  (18/5)
    # if (v<0.09):
    #     return 0 
    # print("Velocity in python - ",v)
    # print("V is ",v)
    return v
# print(velocity(13.846948, 100.565309, 13.847230, 100.565480, 15))
# def findvelocity(lat, lon):
# print(velocity(13.846282, 100.568611, 13.846283, 100.568611, 30))
# print(velocity(13.846282005310059, 100.56861114501953, 13.84628296, 100.5686111, 30))
def direction(lat2, lon2, lat1, lon1):
    rad = atan2(lon2-lon1,lat2-lat1)
    compassReading = rad*(180/math.pi)
    if(compassReading < 0):
        compassReading += 360
    # print(compassReading)
    coordNames = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    coordIndex = round(compassReading / 45)
    if coordIndex < 0:
        coordIndex = coordIndex + 8
    # return coordNames[coordIndex]
    return compassReading
def direction2(lat2, lon2, lat1, lon1):
    rad = atan2(lon2-lon1,lat2-lat1)
    compassReading = rad*(180/math.pi)
    if(compassReading < 0):
        compassReading += 360
    # print(compassReading)
    coordNames = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    coordIndex = round(compassReading / 45)
    if coordIndex < 0:
        coordIndex = coordIndex + 8
    return coordNames[coordIndex]
    # return compassReading
print("Center to 1: ",direction(13.847141263581113, 100.56554275891207	, 13.84704991779301, 100.56554458561267)," Convert to ", direction2(13.847141263581113, 100.56554275891207	, 13.84704991779301, 100.56554458561267))
print("Center to 2: ",direction(13.847124335904912, 100.56562322518144	, 13.84704991779301, 100.56554458561267), " Convert to ", direction2(13.847124335904912, 100.56562322518144	, 13.84704991779301, 100.56554458561267))
print("Center to 3: ",direction(13.847049463476155, 100.56563462456961	, 13.84704991779301, 100.56554458561267), " Convert to ", direction2(13.847049463476155, 100.56563462456961	, 13.84704991779301, 100.56554458561267))
print("Center to 4: ",direction(13.846978970854241, 100.56563760790536	, 13.84704991779301, 100.56554458561267), " Convert to ", direction2(13.846978970854241, 100.56563760790536	, 13.84704991779301, 100.56554458561267))
print("Center to 5: ",direction(13.846979148478031, 100.56554275891203	, 13.84704991779301, 100.56554458561267), " Convert to ", direction2(13.846979148478031, 100.56554275891203	, 13.84704991779301, 100.56554458561267))
print("Center to 6: ",direction(13.846980450607829, 100.56547100982182	, 13.84704991779301, 100.56554458561267), " Convert to ", direction2(13.846980450607829, 100.56547100982182	, 13.84704991779301, 100.56554458561267))
print("Center to 7: ",direction(13.847054671994446, 100.565472350926	, 13.84704991779301, 100.56554458561267), " Convert to ", direction2(13.847054671994446, 100.565472350926	, 13.84704991779301, 100.56554458561267))
print("Center to 8: ",direction(13.847143867839604, 100.56546698650804	, 13.84704991779301, 100.56554458561267), " Convert to ", direction2(13.847143867839604, 100.56546698650804	, 13.84704991779301, 100.56554458561267))
# print(direction(13.84627151, 100.56861115, 13.84627247, 100.56861115))
# print(direction(13.84627247, 100.56861115, 13.84627151, 100.56861115))

def Rec(pitch, roll, yaw):
    # print(type(pitch), pitch)
    # pitch = changetorad(pitch)
    # print(type(pitch),pitch)
    rx = np.array([[1, 0, 0], 
                    [0, cos(roll), sin(roll)],
                    [0, -sin(roll), cos(roll)]])
    # print("\nRx is ",rx)
    ry = np.array([[cos(pitch), 0, sin(pitch)], 
                    [0, 1, 0],
                    [-sin(pitch), 0, cos(pitch)]])
    rz = np.array([[cos(yaw), sin(yaw), 0], 
                    [-sin(yaw), cos(yaw), 0],
                    [0, 0, 1]])
    result = np.dot(rx, ry)
    # print("Test rx*ry ",result)
    result = np.dot(result, rz)
    # result = np.array([[6, 1, 1],
    #                     [4, -2, 5],
    #                     [2, 8, 7]])
    # print(np.linalg.inv(result))
    # print("\nRce-> ",result,"\n")
    # print("Rec inverse->", np.linalg.inv(result))
    # test = np.dot(result,np.linalg.inv(result))
    # print("test is ",test)
    return np.linalg.inv(result)

def Pcimt(focal_length, fx, fy, cx, cy, u, v):
    # print("u v is ",u,v)
    temp1 = np.array([[0, 0, focal_length],
                    [-focal_length, 0, 0],
                    [0, -focal_length, 0]])
    temp2 = np.array([[1/fx, 0, -cx/fx],
                    [0, 1/fy, -cy/fy],
                    [0, 0, 1]])
    temp3 = np.array([[u],
                        [v], 
                        [1]])
    # print("temp1 is ",temp1)
    # print("temp2 is ",temp2)
    # print("temp3 is ",temp3)
    result = np.dot(temp1, temp2)
    # print("result temp1 is ",result)
    result = np.dot(result, temp3)
    # print("Pcimt-> ",result)
    return result

def absPcimt(pcimt):
    # print("absPcimt-> ",pow(pow(pcimt[0], 2) + pow(pcimt[1], 2) + pow(pcimt[2], 2), 1/2))
    return pow(pow(pcimt[0], 2) + pow(pcimt[1], 2) + pow(pcimt[2], 2), 1/2)

def cald(height, rec, pimt, abspimt):
    lr = np.array([[0], #T
                    [0], 
                    [-1] ])
                
    # test = np.array([0,0,-1])
    # print(lr.shape, test.shape)
    # print("lr -> ",lr)
    # print("rec is ",rec," \npimt is ",pimt,"\n abspimt is \n",abspimt, "height is ", height)
    ls = np.dot(rec, pimt)
    ls = ls/abspimt
    # print("ls -> ",ls)
    result = height/(-1*ls[2])
    # print("d-> ",result)
    # ls2 = (np.dot(rec, pimt))*result/abspimt
    # print("LS2 ",ls2)
    # d = height/(-1*ls2[2])
    # print("D is",d)
    # result = -85.81
    # print("Result d ",result)
    # return result 
    # return -85
    return result
    # return -76
    # return -46.7
    # return -40
    # return -71
# print(cald(33.6,Rec(get_pitch()+angle,get_roll(),get_yaw()),np.array(3.67,0.236611,0.54505983),1))
def Pet(rec, d, pcimt, abspcimt):
    result = np.dot(rec, pcimt)*d/abspcimt
    # print("Pet-> ",result)
    return result

def findlat(u, v):
    # print("Find Lat")
    # u = 10
    # v = 10
    rec  = Rec(get_pitch()+angle, get_roll(), get_yaw())
    pimt = Pcimt(f, fx, fy, cx, cy, u, v)
    abspimt = absPcimt(pimt)
    d = cald(height, rec, pimt, abspimt)
    # print(angle, d)
    enu_frame = Pet(rec, d, pimt, abspimt)
    return get_lat(lat_uav(), lon_uav(), height, enu_frame[0], enu_frame[1], enu_frame[2])

def findlon(u, v):
    # u = 10
    # v = 10
    # print("\nFind lon")
    rec = Rec(get_pitch()+angle, get_roll(), get_yaw())
    pimt = Pcimt(f, fx, fy, cx, cy, u, v)
    abspimt = absPcimt(pimt)
    d = cald(height, rec, pimt, abspimt)
    enu_frame = Pet(rec, d, pimt, abspimt)
    return get_lon(lat_uav(), lon_uav(), height, enu_frame[0], enu_frame[1], enu_frame[2])
u = 412
v = 396
lat2 = 13.847478258048547
lon2 = 100.56481686231754
# 13.84623626074483, 100.56851401961089
# print(findlat(u, v), findlon(u, v))
# for i in range(10):
#     rec  = Rec(radians(-1*i*10), get_roll(), get_yaw())
#     pimt = Pcimt(f, fx, fy, cx, cy, u, v)
#     abspimt = absPcimt(pimt)
#     d = cald(height, rec, pimt, abspimt)
#     print(i*10, d)
# print(Distance(findlat(u, v), findlon(u, v), lat2, lon2))
u = 452
v = 337
lat2 = 13.84764510095109
lon2 = 100.56455279460229
# print(findlat(u, v), findlon(u, v))
# print(Distance(findlat(u, v), findlon(u, v), lat2, lon2))
# print(Distance(13.847150,100.565440,13.84764510095109,100.56455279460229))
# print(Distance(lat_uav(), lon_uav(), lat2, lon2))
# f = open('location.txt', 'w')
# x = 0
# for i in range(1024):
#     for j in range(576):
#         print(i ,j)
#         lat = float(findlat(i, j))
#         lon = float(findlon(i, j))
#         print(lat, lon)
#         with open('location.txt', 'w') as f2:
#             f2.write("1 " + str(lat) + " " + str(lon) + " 0 0\n")
#         # f = open('location.txt', 'w')
#         # f.write("1 " + lat + " " + lon + " 0 0\n")
#         # f.close()
#     x = x+100
# f.close()
