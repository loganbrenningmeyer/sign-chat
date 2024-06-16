import os

for dirpath, dirnames, filenames in os.walk('../SignChat_Data/American Sign Language Letters.v1-v1.yolov8/valid/images'):

    # Move all files to their proper folder
    for file in filenames:
        os.rename(os.path.join('../SignChat_Data/American Sign Language Letters.v1-v1.yolov8/valid/images', file),
                  os.path.join(f'../SignChat_Data/American Sign Language Letters.v1-v1.yolov8/valid/{file[0]}', file))