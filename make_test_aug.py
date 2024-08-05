import os
import json
import cv2
import random
import albumentations as A

project_dir = 'annotation-2'

with open(f'projects/{project_dir}/result.json') as json_file:
    data = json.load(json_file)

transform_pipeline = A.Compose([
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=0,
        rotate=0,
        shear=(-3, 3),
        cval=(0, 0, 0),
        p=1
    ),
], bbox_params=A.BboxParams(
    format='coco',
    min_visibility=0.8,
    label_fields=['class_labels']
))

dir = 'test'

if not os.path.exists(dir):
    os.makedirs(dir)

images = []
ignore = set([
    '35a9967a-IMG_2624-77756',
    'aa618e78-IMG_2633-11729',
    '1f7b5e8d-IMG_2597-10312'
])
include = set([
    '1c1f1af8-IMG_5702-67448',
    '5b453f98-IMG_5708-31806',
    '5c784cd0-IMG_5703-87512',
    '41edc22d-IMG_5707-79571',
    '60e97e43-IMG_5704-79170',
    'b6b3aeea-IMG_5709-40473',
])

for annotation in data['annotations']:
    if len(images) < annotation['image_id'] + 1:
        images.append({ 
            'file_name': data['images'][annotation['image_id']]['file_name'],
            'bboxes': [],
            'labels': []
        })

    label = data['categories'][annotation['category_id']]['name']
    
    images[-1]['bboxes'].append(annotation['bbox'])
    images[-1]['labels'].append(label)

for image in images:
    image_name = image['file_name'].split('.')[0].split('/')[1]

    if image_name not in include:
        continue

    img = cv2.imread(f'projects/{project_dir}/{image["file_name"]}')

    for i in range(1, 51):
        transformed = transform_pipeline(image=img, bboxes=image['bboxes'], class_labels=image['labels'])
        transformed_img = transformed['image']
        cv2.imwrite(f'{dir}/eng_{image_name}_aug_{i}.tif', transformed_img)