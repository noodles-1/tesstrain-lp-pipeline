import os
import cv2
import pytesseract

def min_error(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][n] = m - i
    for i in range(n + 1):
        dp[m][i] = n - i
    
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] if str1[i] == str2[j] else min(
                dp[i + 1][j],
                dp[i][j + 1],
                dp[i + 1][j + 1]
            ) + 1
    
    return dp[0][0]

total_err = 0
total = 0
ground_plate = 'GAD3282'

for image_file in os.listdir('test'):
    image = cv2.imread(f'test/{image_file}')

    tesseract_predicted = pytesseract.image_to_string(image, lang='LP', config='--tessdata-dir tesstrain/data --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
    tesseract_predicted = tesseract_predicted.strip().replace(' ', '')
    total_err += min_error(tesseract_predicted, ground_plate)
    total += len(ground_plate)

accuracy = (total - total_err) / total
print(f'accuracy: ', accuracy * 100)