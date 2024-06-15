import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import scipy.io as sio
import numpy as np
import random
import cv2

# -- YOLO
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

'''
Model for hand bounding box estimation
- Options
    * YOLO

Datasets:
- ASL Alphabet
- EgoHands
- Hand Dataset
- Multiview Hand Pose Dataset

Model Input:
- Image
    * Image is of any size, YOLOv5 can take images of different sizes during training
        - YOLOv5 resizes the images to a fixed size using padding to conserve ratio
- Annotations
    * Each image has one .txt file with a single line for each bounding box, the format
    of each row is:
        - class_id center_x center_y width height
    * The coordinates are normalized from [0, 1]
        - To convert to normalized xywh from pixel values:
            * Divide center_x and width by the image's width
            * Divide center_y and height by the image's height

Model Output:
- Bounding box
    * YOLOv5 expects annotations in a .txt file where each line of the text file describes a bounding box
        - class x_center y_center width height
            * normalized [0, 1] for the image's size
        - classes: not hand (0), hand (1)
    * Take bbox annotations from datasets and convert into a .txt file for each image in the
    proper YOLOv5 format
    * 2nd stage classifier will take the bounding boxes and convert them to a fixed size

Handling Datasets:
- Multiview
    * Images            {shot_idx}_webcam_{image_idx}.jpg
        - 480px * 640px
    * Bounding boxes    {shot_idx}_bbox_{image_idx}.txt
        - File:
            * TOP 
            * LEFT
            * BOTTOM
            * RIGHT
        - Convert to YOLO format and save to .txt file:
            * x_center = (LEFT + RIGHT) / 2
            * y_center = (TOP + BOTTOM) / 2
            * width = RIGHT - LEFT
            * height = BOTTOM - TOP

- EgoHands
    * Images            frame_{frame_idx}.jpg 
        - 720px * 1280px
    * Bounding boxes    polygons.mat
        - 
            
'''


'''
Plots bounding box data for multiview dataset
'''
def plot_multiview_bbox(data_idx, shot_idx, image_idx):
    '''
    Set the paths to the image and its respective bounding box file
    '''
    image = Image.open(f"../SignChat_Data/multiview_hand_pose_dataset_release/data_{data_idx}/{shot_idx}_webcam_{image_idx}.jpg")
    bbox_file = f"{shot_idx}_bbox_{image_idx}.txt"

    # -- Create plot and show image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # -- Read bounding box file
    with open(f"../SignChat_Data/multiview_hand_pose_dataset_release/data_{data_idx}/{bbox_file}") as f:
        # -- Read the bounding box values
        top = int(str.split(f.readline())[1])
        left = int(str.split(f.readline())[1])
        bottom = int(str.split(f.readline())[1])
        right = int(str.split(f.readline())[1])
        
        width = right - left
        height = bottom - top

        # -- Create a Rectangle patch
        rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    print(top, left, bottom, right)

    plt.axis('off')
    plt.show()

'''
Plots the image and bounding boxes for the 'frame_idx' image [0, 99] within the 'sample_path' directory
'''
def plot_egohands_bbox(sample_path, frame_idx):
    '''
    Read polygon data
    '''
    polygons = sio.loadmat(os.path.join(sample_path, 'polygons.mat'))['polygons'][0]

    '''
    Read through each .jpg image in the sample folder
    '''
    for dirpath, dirnames, filenames in os.walk(sample_path):
        filenames = sorted([f for f in filenames if 'jpg' in str.split(f, '.')])
        break # Only read top directory

    # -- Read image
    image = Image.open(os.path.join(sample_path, filenames[frame_idx]))

    # -- Create plot and show image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    '''
    polygon_array is an array of hand bounding data 
    for 100 frames from the respective video

    The values for each frame in polygon_array is 4 arrays
    for "own left", "own right", "other left", "other right"

    Each hand has ~100 (x, y) coordinates defining their bounds

    Polygon array for each hand is the complete shape of the hand
    To find the bounding box, need to find the min_x, max_x, min_y, and max_y

    hands:
    - [own_left, own_right, other_left, other_right]
    - hands_labels = ["own_left", "own_right", "other_left", "other_right"]
    '''
    hands = [polygons[frame_idx][0], polygons[frame_idx][1], polygons[frame_idx][2], polygons[frame_idx][3]]

    '''
    bbox:
    - {'hand_label': [min_x, max_x, min_y, max_y]}
    '''
    hands_bbox = {'own_left': [], 'own_right': [], 'other_left': [], 'other_right': []}

    '''
    Calculate bounding boxes and plot rectangle patches
    '''
    for hand_label, hand in zip(hands_bbox.keys(), hands):
        # -- Hand with no bbox data
        if len(hand[0]) == 0:
            hands_bbox[hand_label] = []
        else:
            # -- Calculate bounding box data        
            min_x, max_x = np.min(hand.T[0]), np.max(hand.T[0])
            min_y, max_y = np.min(hand.T[1]), np.max(hand.T[1])
            # -- Store the bounding box with the proper hand
            hands_bbox[hand_label] = [min_x, max_x, min_y, max_y]

            '''
            Plot rectangle patch
            '''
            # -- Set parameters (left, top, width, height)
            left = min_x
            top = min_y
            width = max_x - min_x
            height = max_y - min_y

            # -- Create a Rectangle patch
            rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    # -- Display plot
    plt.show()


'''
Create datasets for YOLO model

YOLO data directory:
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   └── val/
│       ├── img1.jpg
│       ├── img2.jpg
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   └── val/
│       ├── img1.txt
│       ├── img2.txt


Image:
- .jpg or .png
- YOLO resizes images automatically during training/inference

Annotations:
- .txt
- Per bounding box:
    * class_id x_center y_center width height
    * hand class_id = 0

Params:
- dataset {0: both, 1: Multiview, 2: EgoHands}
'''
def create_datasets(dataset=0, shuffle=True):
    if dataset not in range(3):
        return None
    
    '''
    Group all images into a list and all annotations into a list 
    in their respective order

    images = [image1, image2, ...] (image = Image.open().convert('RGB'))
    annotations = [[x_center1, y_center1, width1, height1], ...] (normalized [0, 1] by original image dimensions)
    '''
    images = []
    annotations = []

    if (dataset == 1 or dataset == 0):
        '''
        Multiview Dataset

        data_idx: [1, 21]
        '''
        # -- Iterate through each data folder
        for data_idx in range(1, 2):
            # -- Set data directory
            data_dir = f"../SignChat_Data/multiview_hand_pose_dataset_release/data_{data_idx}"

            # -- Iterate through all files in data_dir
            for dirpath, dirnames, filenames in os.walk(data_dir):
                # -- Get all image filenames
                image_filenames = sorted([f for f in filenames if 'jpg' in str.split(f, '.')])

                for i, image_file in enumerate(image_filenames):
                    # -- Read the image and store in images array
                    with Image.open(os.path.join(data_dir, image_file)) as image:
                        images.append(image.convert('RGB'))

                    # if i % 50 == 0:
                    #     print(f"Filename: {os.path.join(data_dir, image_file)}")
                    #     os.system('cls')

                    bbox_file = f"{str.split(image_file, '_')[0]}_bbox_{str.split(str.split(image_file, '_')[2], '.')[0]}.txt"

                    # -- Convert the bounding box coordinates to YOLO format and store in annotations array
                    with open(os.path.join(data_dir, bbox_file)) as f:
                        # -- Read the bounding box values
                        top = int(str.split(f.readline())[1])
                        left = int(str.split(f.readline())[1])
                        bottom = int(str.split(f.readline())[1])
                        right = int(str.split(f.readline())[1])

                        # -- Calculate YOLO annotation
                        x_center = (left + right) / 2
                        y_center = (top + bottom) / 2
                        width = right - left
                        height = bottom - top

                        # -- Normalize by image dimensions (480, 640)
                        x_center /= 640.0
                        width /= 640.0
                        y_center /= 480.0
                        height /= 480.0

                        # -- Store the annotation in annotations
                        annotations.append([[0, x_center, y_center, width, height]])

    if (dataset == 2 or dataset == 0):
        '''
        EgoHands Dataset
        '''
        # -- Set data directory
        data_dir = '../SignChat_Data/egohands_data/_LABELLED_SAMPLES'

        # -- Get all datafolders
        data_folders = []
        for dirpath, dirnames, filenames in os.walk(data_dir):
            data_folders = dirnames
            break

        # -- Iterate through all datafolders
        for folder in data_folders:
            for dirpath, dirnames, filenames in os.walk(os.path.join(data_dir, folder)):
                # -- Get all image filenames
                image_filenames = sorted([f for f in filenames if 'jpg' in str.split(f, '.')])

                # -- Read the images and store in images array
                for image_file in image_filenames:
                    with Image.open(os.path.join(data_dir, folder, image_file)) as image:
                        images.append(image.convert('RGB'))

                '''
                polygons is an array of hand bounding data 
                for 100 frames from the respective video

                The values for each frame in polygon_array is 4 arrays
                for "own left", "own right", "other left", "other right"

                Each hand has ~100 (x, y) coordinates defining their bounds

                Polygon array for each hand is the complete shape of the hand
                To find the bounding box, need to find the min_x, max_x, min_y, and max_y

                hands:
                - [own_left, own_right, other_left, other_right]
                - hands_labels = ["own_left", "own_right", "other_left", "other_right"]
                '''
                # -- Read polygons.mat file
                polygons = sio.loadmat(os.path.join(data_dir, folder, 'polygons.mat'))['polygons'][0]

                for frame_idx in range(len(image_filenames)):
                    hands = [polygons[frame_idx][0], polygons[frame_idx][1], polygons[frame_idx][2], polygons[frame_idx][3]]

                    # -- Store multiple bounding boxes per frame
                    frame_annotations = []

                    # -- Calculate bounding box data for each hand
                    for hand in hands:       
                        # -- Ignore if there is no bounding box data
                        if len(hand) <= 1:
                            continue
                            
                        min_x, max_x = np.min(hand.T[0]), np.max(hand.T[0])
                        min_y, max_y = np.min(hand.T[1]), np.max(hand.T[1])

                        # -- Calculate YOLO annotation
                        x_center = (min_x + max_x) / 2
                        y_center = (min_y + max_y) / 2
                        width = max_x - min_x
                        height = max_y - min_y

                        # -- Normalize by image dimensions (720, 1280)
                        x_center /= 1280.0
                        width /= 1280.0
                        y_center /= 720.0
                        height /= 720.0

                        # -- Add to frame annotations
                        frame_annotations.append([0, x_center, y_center, width, height])

                    # -- Store frame's annotations in annotations array
                    annotations.append(frame_annotations)

    '''
    Prepare training/testing datasets
    '''
    # -- Shuffle arrays in order
    if shuffle:
        combined = list(zip(images, annotations))
        random.shuffle(combined)
        images, annotations = zip(*combined)
        images = list(images)
        annotations = list(annotations)

    print(len(images))
    print(len(annotations))

    # -- Create training/testing splits
    train_images = [image for image in images[:int(len(images) * 0.8)]]
    val_images = [image for image in images[int(len(images) * 0.8):]]

    train_annotations = annotations[:int(len(annotations) * 0.8)]
    val_annotations = annotations[int(len(annotations) * 0.8):]


    '''
    Save image files into dataset folder
    '''
    for i, (train_image, train_annotation) in enumerate(zip(train_images, train_annotations)):
        # -- Save training image
        train_image.save(f'dataset/images/train/image{i}.jpg')
        # -- Save training annotation
        annotation_txt = [' '.join(map(str, annotation)) for annotation in train_annotation]
        with open(f'dataset/labels/train/image{i}.txt', 'w') as file:
            for annotation_line in annotation_txt:
                file.write(annotation_line + '\n')

    for i, (val_image, val_annotation) in enumerate(zip(val_images, val_annotations)):
        # -- Save validation image
        val_image.save(f'dataset/images/val/image{i}.jpg')
        # -- Save validation annotation
        annotation_txt = [' '.join(map(str, annotation)) for annotation in val_annotation]
        with open(f'dataset/labels/val/image{i}.txt', 'w') as file:
            for annotation_line in annotation_txt:
                file.write(annotation_line + '\n')

    '''
    Return:
    - images (array of PIL.Image files)
    - annotations (array of [class_id, x_center, y_center, width, height] bbox data)
    '''
    return train_images, train_annotations, val_images, val_annotations

'''
Train YOLO model
'''
def train_YOLO():
    model = YOLO('yolov8n.yaml')

    model.train(data='box.yaml', epochs=100, imgsz=640, batch=16, device=0, workers=8)

    metrics = model.val()

if __name__ == "__main__":


    model = YOLO('runs/detect/train5/weights/best.pt')

    # -- Create video capture window
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        _, img = cap.read()

        results = model.predict(img)

        for r in results:

            annotator = Annotator(img)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, model.names[int(c)])

        img = annotator.result()
        cv2.imshow('YOLO V8 Detection', img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    results = model('sample_img.jpg')
    
    print(results.boxes)

    '''
    Train YOLOv5 model on images and annotations
    '''


    
