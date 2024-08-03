import json
import cv2
import random
import albumentations as A

project_dir = 'demo-project-4'

with open(f'projects/{project_dir}/result.json') as json_file:
    data = json.load(json_file)

scale_pipeline = A.Compose([
    A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_LINEAR),
    A.PadIfNeeded(
        min_height=300,
        min_width=640,
        border_mode=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
], bbox_params=A.BboxParams(
    format='coco',
    label_fields=['class_labels']
))

transform_pipeline = A.Compose([
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(0.05, 0.05),
        rotate=0,
        shear=(-2, 2),
        cval=(0, 0, 0),
        p=1
    ),
    A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_LINEAR),
    A.PadIfNeeded(
        min_height=300,
        min_width=640,
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
], bbox_params=A.BboxParams(
    format='coco',
    min_visibility=0.8,
    label_fields=['class_labels']
))

dir = 'tesstrain/data/LP-ground-truth'
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

    for i in range(1, 5):
        scaled = scale_pipeline(image=img, bboxes=image['bboxes'], class_labels=image['labels'])
        scaled_img = scaled['image']
        scaled_bboxes = scaled['bboxes']
        scaled_labels = scaled['class_labels']

        image_name = image['file_name'].split('.')[0].split('/')[1]

        with open(f'{dir}/eng_{image_name}_{i}.gt.txt', mode='w') as file:
            file.write(''.join(scaled_labels))

        with open(f'{dir}/eng_{image_name}_{i}.box', mode='a') as file:
            for j in range(len(scaled_bboxes)):
                x, y, w, h = scaled_bboxes[j]
                bbox = [int(x), int(y), int(x + w), int(y + h)]

                line = [scaled_labels[j]]
                line.extend(bbox)
                line.extend(['0', '\n'])
                line = [str(elem) for elem in line]
                file.write(' '.join(line))
        
        cv2.imwrite(f'{dir}/eng_{image_name}_{i}.tif', scaled_img)

    for i in range(1, 5):
        transformed = transform_pipeline(image=img, bboxes=image['bboxes'], class_labels=image['labels'])
        transformed_img = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']

        with open(f'{dir}/eng_{image_name}_aug_{i}.gt.txt', mode='w') as file:
           file.write(''.join(transformed_labels))

        with open(f'{dir}/eng_{image_name}_aug_{i}.box', mode='a') as file:
            for j in range(len(transformed_bboxes)):
                x, y, w, h = transformed_bboxes[j]
                bbox = [int(x), int(y), int(x + w), int(y + h)]

                line = [transformed_labels[j]]
                line.extend(bbox)
                line.extend(['0', '\n'])
                line = [str(elem) for elem in line]
                file.write(' '.join(line))

        cv2.imwrite(f'{dir}/eng_{image_name}_aug_{i}.tif', transformed_img)