! pip install numpy
! pip install pandas
! pip install matplotlib
! pip install opencv-python
! pip install tqdm
! pip install pyyaml
! pip install torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import yaml
import torch

annotations = pd.read_csv('/content/drive/MyDrive/test/data/train_solution_bounding_boxes (1).csv')
annotations.head()

base_path = '/content/drive/MyDrive/test/data/training_images'
fig, ax = plt.subplots(figsize=(16,16), nrows=4, ncols=4)
nrows=4
ncols=4
indices = np.random.randint(low=0, high=len(annotations), size=(nrows, ncols))
for i in range(nrows):
    for j in range(ncols):
        idx = indices[i,j]
        row = annotations.loc[idx]
        start = (int(row['xmin']), int(row['ymin']))
        end = (int(row['xmax']), int(row['ymax']))
        img_path = os.path.join(base_path, row['image'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.rectangle(img, start, end, color=(0,255,0), thickness=2)
        ax[i,j].axis('off')
        ax[i,j].imshow(img)
plt.tight_layout()
plt.show()

annotations['xcenter'] = (annotations['xmin']+annotations['xmax'])/2
annotations['ycenter'] = (annotations['ymin']+annotations['ymax'])/2
annotations['width'] = (annotations['xmax'] - annotations['xmin'])
annotations['height'] = (annotations['ymax'] - annotations['ymin'])

h,w,c = img.shape

annotations['xcenter'] = annotations['xcenter']/w
annotations['width'] = annotations['width']/w
annotations['ycenter'] = annotations['ycenter']/h
annotations['height'] = annotations['height']/h
annotations['image'] = annotations['image'].apply(lambda x: os.path.join(base_path,x))
annotations = annotations[['image', 'xcenter','ycenter','width','height']]

annotations.head()

train_path = os.path.join('Data','train')
test_path = os.path.join('Data','test')
val_path = os.path.join('Data','val')

os.makedirs(train_path, exist_ok=True)
print('Made folder for train images')
os.makedirs(val_path, exist_ok=True)
print('Made folder for validation images')
os.makedirs(test_path, exist_ok=True)
print('Made folder for test images')

test_size = int(0.1 * len(annotations))
train_df, test_df = train_test_split(annotations, test_size=test_size)
train_df, val_df = train_test_split(train_df, test_size=test_size)

data = {
    'train':{
        'df':train_df,
        'path':train_path
    },
    'val':{
        'df':val_df,
        'path':val_path
    },
    'test':{
        'df':test_df,
        'path':test_path
    }
}

for subset, subset_data in data.items():
    df = subset_data['df']
    dst_path = subset_data['path']
    for idx, row in tqdm(df.iterrows()):
        src = row['image']
        img_id = src.split(os.path.sep)[-1].split('.')[0]
        extension = src.split(os.path.sep)[-1].split('.')[1]
        img_dst = os.path.join(dst_path, f'{img_id}.{extension}')
        shutil.copy2(src, img_dst)
        annotation_text = f"0 {row['xcenter']} {row['ycenter']} {row['width']} {row['height']}"
        with open(os.path.join(dst_path, f'{img_id}.txt'), 'w+') as f:
            f.write(annotation_text)
            f.write('\n')
    print(f'Done moving files for {subset} set')

config = {
    'names':['Car'],
    'nc':1,
    'train':os.path.abspath(train_path),
    'val':os.path.abspath(val_path),
    'test':os.path.abspath(test_path)
}

with open('data.yaml', 'w+') as f:
    yaml.dump(config, f, default_flow_style=False)

! git clone https://github.com/ultralytics/yolov5.git

! pip install -r yolov5/requirements.txt

! wandb disabled

! python yolov5/train.py --data data.yaml --img 640 --batch 16 --epochs 20 --weights yolov5/yolov5s.pt

!apt-get install -y git

%cd yolov5

  !python detect.py --source /content/Data/test/vid_4_13580.jpg
  !python detect.py --source /content/Data/test/vid_4_13660.jpg
  !python detect.py --source /content/yolov5/Data/test/vid_4_12300.jpg
