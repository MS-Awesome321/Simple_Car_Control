import cv2
from ultralytics import YOLO, YOLOE
from PIL import ImageGrab
import numpy as np
from scipy.ndimage import center_of_mass
import pyautogui as pag
import time

# Load the YOLO11 model
# model = YOLO("/Users/mayanksengupta/Desktop/CV_Training/runs/segment/train3/weights/best.pt")

model = YOLOE("yoloe-11l-seg.pt")
names = ["car", "road"]
model.set_classes(names, model.get_text_pe(names))

# Loop through the video frames
try:
    moveYOLOWindow = True
    moveMaskWindow = True
    i = 0
    interval = 1

    while True:
        # Read a frame from the video
        frame = np.array(ImageGrab.grab(bbox=(42, 130, 553, 834)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        delta_x = 200

        if frame is not None:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.5)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                class_indices = results[0].boxes.cls.cpu().numpy()

                if masks.shape[0] == 2:
                    combined_mask = np.logical_or(masks[0], masks[1])
                    combined_mask = combined_mask.astype('uint8')

                    show = 255 * combined_mask
                    show = np.stack([show, show, show], axis=-1)
                    
                    cm_car = np.round(center_of_mass(masks[class_indices == 0][0])).astype(int)
                    print(cm_car)
                    
                    cv2.circle(show, tuple(cm_car[::-1]), 3, (255,0,0), -1)
                    
                    for y in range(combined_mask.shape[0]):
                        if not (combined_mask[y, :] == 0).all():
                            x = np.flatnonzero(combined_mask[y, :]).mean()
                            x = int(x)
                            show[y, x, :] = np.array([0, 0, 255])
                            if y == cm_car[0]:
                                cv2.circle(show, (x, y), 3, (0, 255, 0), -1)
                                delta_x = cm_car[0] - x
                                
                            
                    cv2.imshow("Road Mask", show)
                    if moveMaskWindow:
                        cv2.moveWindow("Road Mask", 1225, 0)
                        moveMaskWindow=False

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            if moveYOLOWindow:
                cv2.moveWindow("YOLO11 Tracking", 700, 0)
                moveYOLOWindow=False

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop video feed is cut
            break

        # Control the car
        print(delta_x)
        if delta_x > 200:
            pag.keyDown('a')
            # if i % interval == 0:
            #     pag.keyDown('w')
            time.sleep(0.00005 * abs(delta_x - 200)**2)
            pag.keyUp('a')
            # time.sleep(0.1)
            # if i % interval == 0:
            #     pag.keyUp('w')
        else:
            pag.keyDown('d')
            # if i % interval == 0:
            #     pag.keyDown('w')
            time.sleep(0.00005 * abs(delta_x - 200)**2)
            pag.keyUp('d')
            # time.sleep(0.1)
            # if i % interval == 0:
            #     pag.keyUp('w')
       
        i += 1

        if i in [5, 10, 20, 40, 80]:
            interval += 1

except pag.FailSafeException:
    print('Failsafe Triggered, Ending program')

# Lift control keys up  
pag.FAILSAFE = False 
pag.keyUp('a')
pag.keyUp('w')
pag.keyUp('d')
pag.FAILSAFE = True 

# Close the display window
cv2.destroyAllWindows()