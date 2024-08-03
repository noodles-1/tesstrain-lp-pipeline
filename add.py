import random

def generate_int():
    length = random.randint(30, 40)  # Random length between 30 and 40
    characters = '0123456789 '       # Possible characters (numbers and space)
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

for _ in range(10000):
    with open('langdata/eng.training_text', mode='a') as file:
        file.write(generate_int() + '\n')