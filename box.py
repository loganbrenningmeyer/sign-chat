import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import scipy.io as sio
import numpy as np
# from oct2py import Oct2Py
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
Read data into training/testing datasets
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



if __name__ == "__main__":
    '''
    Plot multiview bounding box data
    '''
    for data_idx in range(3, 5):
        for shot_idx in range(2):
            for image_idx in range(3, 5):
                plot_multiview_bbox(data_idx, shot_idx, image_idx)

    '''
    Plot egohands bounding box data
    '''
    for i in range(10):
        plot_egohands_bbox('../SignChat_Data/egohands_data/_LABELLED_SAMPLES/CHESS_COURTYARD_H_S', i)



    
