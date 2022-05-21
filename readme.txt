#### SENDER ####

# TCP
file: sk2_send.py 
command: python3 sk2_send.py {Your own ip}
description: Run this program on sender (Jetson Nano)


#### RECEIVER ####
file: yolo_2socket.py
command: python3 yolo_2socket.py --output ./output/sk2_cam2_1.avi --yolo ./yolo-coco -ip 192.x.x.x # sender's ip

file: ts.py 
description: there are function to calculate geoloation 
