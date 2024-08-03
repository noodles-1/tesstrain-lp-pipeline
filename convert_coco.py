import json
import cv2

project_dir = 'project-1'

with open(f'{project_dir}/result.json') as json_file:
    data = json.load(json_file)

data['annotations'].append({ 'image_id': -1 })
    
image_id = 0
plate = []
for annotation in data['annotations']:
    if annotation['image_id'] != image_id:
        actual_plate = ''.join(plate)
        actual_file = data['images'][image_id]['file_name']
        file_name = actual_file.split('/')[1].split('.')[0]

        with open(f'tesstrain/data/LP-ground-truth/eng_{file_name}.gt.txt', mode='w') as file:
            file.write(actual_plate)

        img = cv2.imread(f'{project_dir}/{actual_file}')
        cv2.imwrite(f'tesstrain/data/LP-ground-truth/eng_{file_name}.tif', img)

        plate = []
        image_id = annotation['image_id']

    if annotation['image_id'] != -1:
        ch = data['categories'][annotation['category_id']]['name']
        plate.append(ch)
        file_name = data['images'][annotation['image_id']]['file_name'].split('/')[1].split('.')[0]
        x_min, y_min, width, height = annotation['bbox']
        bbox = [str(int(x_min)), str(int(y_min)), str(int(x_min + width)), str(int(y_min + height))]

        with open(f'tesstrain/data/LP-ground-truth/eng_{file_name}.box', mode='a') as file:
            line = [ch]
            line.extend(bbox)
            line.append('0')
            file.write(' '.join(line) + '\n')