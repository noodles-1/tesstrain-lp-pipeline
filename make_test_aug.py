import os
import json
import cv2
import random
import albumentations as A

project_dir = 'annotation-1'

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

for annotation in data['annotations']:
    if len(images) < annotation['image_id'] + 1:
        images.append({ 
            'file_name': data['images'][annotation['image_id']]['file_name'],
            'bboxes': [],
            'labels': []
        })

    label = data['categories'][annotation['category_id']]['name']
    
    images[-1]['bboxes'].append(annotation['bbox'])
    images[-1]['labels'].append(label if label != 'space' else ' ')

for image in images:
    img = cv2.imread(f'projects/{project_dir}/{image["file_name"]}')
    image_name = image['file_name'].split('.')[0].split('/')[1]

    for i in range(1, 51):
        transformed = transform_pipeline(image=img, bboxes=image['bboxes'], class_labels=image['labels'])
        transformed_img = transformed['image']
        cv2.imwrite(f'{dir}/eng_{image_name}_aug_{i}.tif', transformed_img)