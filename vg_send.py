# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import NetGear
import cv2
import time, sys

# define and start the stream on first source ( For e.g #0 index device)
stream1 = CamGear(source=2, logging=True).start() 
options = {"multiserver_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.xxx.x.xxx' with yours !!!
host = sys.argv[1]
server = NetGear(

    address= host, port="5566", protocol="tcp", pattern=1, **options
)
# define and start the stream on second source ( For e.g #1 index device)
#stream2 = CamGear(source=1, logging=True).start() 

# infinite loop
while True:

    frameA = stream1.read()
    # read frames from stream1

    #frameB = stream2.read()
    # read frames from stream2

    # check if any of two frame is None
    if frameA is None :
        #if True break the infinite loop
        break

    # do something with both frameA and frameB here
    cv2.imshow("Output Frame1", frameA)
    #cv2.imshow("Output Frame2", frameB)
    # Show output window of stream1 and stream 2 separately
    server.send(frameA)
    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break

    if key == ord("w"):
        #if 'w' key-pressed save both frameA and frameB at same time
        cv2.imwrite("Image-1.jpg", frameA)
        #cv2.imwrite("Image-2.jpg", frameB)
        #break   #uncomment this line to break out after taking images

cv2.destroyAllWindows()
# close output window

# safely close both video streams
stream1.stop()
server.close()
#stream2.stop()
