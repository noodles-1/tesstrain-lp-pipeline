import cv2
import pytesseract

image_file = '6-30633.jpg'

image = cv2.imread(f'test/{image_file}')

tesseract_predicted = pytesseract.image_to_string(image, lang='eng', config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
tesseract_predicted = tesseract_predicted.strip().replace(' ', '')
print(image_file, tesseract_predicted)