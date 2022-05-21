# SOCKET TCP

import socket
import cv2
import pickle
import struct
import imutils

# Server socket
# create an INET, STREAMing socket
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:',host_ip)
host_ip = 'xxx.xxx.xxx.xxx' #---
port = 10050

port1 = 5000
port2 = 5001
socket_address = (host_ip,port)
print('Socket created')
# bind the socket to the host. 
#The values passed to bind() depend on the address family of the socket
server_socket.bind(socket_address)
print('Socket bind complete')
#listen() enables a server to accept() connections
#listen() has a backlog parameter. 
#It specifies the number of unaccepted connections that the system will allow before refusing new connections.
server_socket.listen(5)
print('Socket now listening')

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    client_socket,addr = server_socket.accept()
    print('Connection from:',addr)

    lat = 0.0
    lon = 0.0
    pit = -0.028415044769644737
    yaw = 1.5417470932006836
    roll = 0.023999786004424095

    frame_info = {  "lat":lat, 
                    "lon":lon, 
                    "pit":pit, 
                    "yaw":yaw, 
                    "roll":roll}
        
    if client_socket:
        vid = cv2.VideoCapture(0)
        while(vid.isOpened()):
            img,frame = vid.read()
            a = pickle.dumps(frame)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)

            #------------------------------------------------------------
            sock.sendto(pickle.dumps(frame_info), (host_ip, port1))
            #print(pickle.dumps(frame_info))
            sock2.sendto(pickle.dumps(frame_info), (host_ip, port2))

            cv2.imshow('Sending...',frame)
            key = cv2.waitKey(1) 
            if key ==13:
                client_socket.close()


