import cv2
import threading
import time
import queue
import scipy.io as sio

'''
Need to:
- Train model to find bounding box around hand 
    * Need dataset with labeled bounding boxes for hand position and/or sign language
    * Models
        - YOLO
        - Selective Search (R-CNN)
        - Lots more
- Train model to classify signed letters using chips
    * Need dataset with labeled letters for cropped photos of hands
    * Models
        - ResNet/ResNeXt
        - R-CNN
        - ConvNet/ConvNeXt
        - Lots lots more
'''

# Thread-safe queue to share frames
frame_queue = queue.Queue(maxsize=10)

def video_output():
    
    vc = cv2.VideoCapture(0)

    if not vc.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = vc.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Put the frame into the frame_queue
        if not frame_queue.full():
            frame_queue.put(frame)

        if cv2.waitKey(1) == 27: # Exit on ESC
            break

    vc.release()

if __name__ == "__main__":
    cv2.namedWindow("preview")

    # Create and start video output thread to run in the background
    video_thread = threading.Thread(target=video_output)
    video_thread.daemon = True
    video_thread.start()

    matfile = sio.loadmat("../SignChat_Data/Hand Dataset/hand_dataset/training_dataset/training_data/annotations/Buffy_1.mat")
    for box in matfile['boxes'][0]:
        print(box)

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Display frame in the preview window
            cv2.imshow("preview", frame)

            # Save the frame to disk
            cv2.imwrite(f"frames/frame.jpg", frame)
        else:
            time.sleep(0.01)
        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    print("Program terminated gracefully.")