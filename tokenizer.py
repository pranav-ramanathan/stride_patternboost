from tokenizers import Tokenizer
from tokenizers.models import BPE
import os
import argparse
import json

from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', '-i', type=str, default='V-input.txt', help="input file with things one per line")
parser.add_argument('--ntokens', type=int, default=100, help="number of tokens to tokenize to")
args = parser.parse_args()

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=args.ntokens)

# Path for the source file and the destination file
source_file_path = args.input_file
destination_file_path = "temp.txt"

print(f'Created {destination_file_path} and training tokenizer...')

# Reading the first 100,000 lines from the source file and training the tokenizer on them
with open(source_file_path, 'r') as source_file, open(destination_file_path, 'w') as destination_file:
    for i in range(100_000):
        line = source_file.readline()
        if not line:
            break
        destination_file.write(line)

directory_name = "tokenizer_data"

if not os.path.isdir(directory_name):
    # Create the directory
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created.")
tokenizer.train([destination_file_path], trainer)
tokenizer.save(directory_name + "/tokenizer.json")

if os.path.exists(destination_file_path):
    os.remove(destination_file_path)
    print(f"File '{destination_file_path}' has been deleted.")

input_file_path = source_file_path
with open(input_file_path, "r") as file:
    text_data = [line.strip() for line in file]

# now create tokenized output file
token_file_out = source_file_path.rsplit('.',1)[0] + '-tokenized.txt'

with open(token_file_out, "w") as file:
    print("Tokenizing training set...")
    for i, sequence in enumerate(text_data):
        if i % 10000 == 0:
            print(f"{i} / {len(text_data)}")
        myids = tokenizer.encode(sequence).ids
        file.write(','.join(["V"+str(id) for id in myids]))
        file.write("\n")

