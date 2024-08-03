import re

def clean_line(line) -> str:
    return re.sub(r'[^a-zA-Z0-9 ]', '', line)

def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'a') as file:
        for line in lines:
            file.write(clean_line(line).upper() + '\n')

input_file = 'langdata/temp.training_text'
output_file = 'langdata/eng.training_text'
process_file(input_file, output_file)