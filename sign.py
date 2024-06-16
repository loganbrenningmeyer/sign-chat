import cv2
import threading
import time
import queue
import scipy.io as sio
import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import datasets, transforms, models

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from classifier.classifier import SignResNeXt

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
    # cv2.namedWindow("preview")

    # # Create and start video output thread to run in the background
    # video_thread = threading.Thread(target=video_output)
    # video_thread.daemon = True
    # video_thread.start()

    # matfile = sio.loadmat("../SignChat_Data/Hand Dataset/hand_dataset/training_dataset/training_data/annotations/Buffy_1.mat")
    # for box in matfile['boxes'][0]:
    #     print(box)

    # while True:
    #     if not frame_queue.empty():
    #         frame = frame_queue.get()

    #         # Display frame in the preview window
    #         cv2.imshow("preview", frame)

    #         # Save the frame to disk
    #         cv2.imwrite(f"frames/frame.jpg", frame)
    #     else:
    #         time.sleep(0.01)
        
    #     if cv2.waitKey(1) == 27:
    #         break

    # cv2.destroyAllWindows()
    # print("Program terminated gracefully.")

    # -- Load Hand Detector
    detector = YOLO('hand_detector/runs/detect/train6/weights/best.pt')

    # -- Load sign classifier
    classifier = SignResNeXt.load_from_checkpoint('classifier/lightning_logs/version_16/checkpoints/epoch=4-step=12425.ckpt')

    # train_YOLO()

    # -- Create video capture window
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        _, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)

        results = detector.predict(img)

        for r in results:

            annotator = Annotator(img)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]

                # c = box.cls
                # -- Crop using bounding box
                x_min = b[0].float()
                y_min = b[1].float()
                x_max = b[2].float()
                y_max = b[3].float()

                # print(x_min, y_min, width, height)

                bbox_img = img[int(y_min) : int(y_max), int(x_min) : int(x_max)]

                # -- Classify cropped image
                bbox_img = Image.fromarray(bbox_img)

                cv2.imshow('BBOX', np.array(cv2.cvtColor(np.array(bbox_img), cv2.COLOR_RGB2BGR)))

                # -- Define transformations
                transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                
                bbox_img = transform(bbox_img)

                # -- Add a batch dimension (since the model expects batches)
                bbox_img = bbox_img.unsqueeze(0)

                print(bbox_img.shape)

                class_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
                
                # -- Pass the bbox_img to the model
                with torch.no_grad():
                    outputs = classifier(bbox_img)

                    print(np.array(outputs).shape)
                    print(f"Outputs: {outputs[0]}\n")

                    probabilities = F.softmax(outputs, dim=1)

                    predicted_class = torch.argmax(outputs, dim=1)

                    print(f"Predicted class: {predicted_class}")

                    predicted_class_idx = predicted_class.item()

                annotator.box_label(box.xyxy[0], f'{predicted_class_idx, (list(class_to_idx.keys())[predicted_class_idx])} : {probabilities[0][predicted_class_idx]}')

        img = annotator.result()
        cv2.imshow('YOLO V8 Detection', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()