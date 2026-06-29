import random
import os

def generate_sample(input_path, output_path, sample_size=100):
    print(f"Reading from {input_path}...")
    sample = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < sample_size:
                sample.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sample[j] = line
                    
    print(f"Writing {len(sample)} candidates to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in sample:
            f.write(line)
            
    print("Done!")

if __name__ == '__main__':
    input_file = 'data/candidates.jsonl'
    output_file = 'data/sample_candidates.jsonl'
    generate_sample(input_file, output_file, 100)
