import argparse
import re

def process_line(line):
    pattern = r'\[(\d+)\] Train loss: (\d+\.\d+) Test loss: (\d+\.\d+) Train Accuracy: (\d+\.\d+) Test Accuracy: (\d+\.\d+)'
    match = re.match(pattern, line)
    if match:
        num_a, num_b, num_c, num_d, num_e = match.groups()
        return f'{num_b}, {num_c}, {num_d}, {num_e}\n'
    else:
        return None

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith('['):
                processed_line = process_line(line)
                if processed_line:
                    outfile.write(processed_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a ResNet on CIFAR10')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)
