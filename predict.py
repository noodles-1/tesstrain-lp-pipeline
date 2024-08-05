import cv2
import pytesseract

image_file = 'eng_de643a8d-IMG_2553-93341_aug_2.tif'

image = cv2.imread(f'test/{image_file}')

tesseract_predicted = pytesseract.image_to_string(image, lang='LP', config='--tessdata-dir tesstrain/data --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
tesseract_predicted = tesseract_predicted.strip().replace(' ', '')
print(image_file, tesseract_predicted)