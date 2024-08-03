import pathlib
import unicodedata
from PIL import Image

red = '\033[91m'
reset = '\033[0m'

def generate_line_box(gt_txt, image_path, output_path):
    """Creates tesseract box files for given (line) image text pairs"""
    lines=pathlib.Path(gt_txt).read_text(encoding='utf-8').splitlines()
    if  len(lines) != 1:
        print(f"{red}Invalid gt_txt file: {gt_txt}{reset}")
        return False
    line = unicodedata.normalize('NFC', lines[0].strip())
    if not line:
        print(f"{red}Can not normalize line in gt_txt file: {blue}{gt_txt}{reset}")
        return False
    with Image.open(image_path) as image:
        width, height = image.size
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for i in range(1, len(line)):
            char = line[i]
            prev_char = line[i-1]
            if unicodedata.combining(char):
                out_file.write(f"{prev_char + char} 0 0 {width} {height} 0\n")
            elif not unicodedata.combining(prev_char):
                out_file.write(f"{prev_char} 0 0 {width} {height} 0\n")
        if not unicodedata.combining(line[-1]):
            out_file.write(f"{line[-1]} 0 0 {width} {height} 0\n")
        out_file.write(f"\t 0 0 {width} {height} 0\n")
    return True