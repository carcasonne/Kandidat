import os
import shutil

# Set your working directory here
fake_key_file = os.path.join("../../keys/fake")
bonafide_key_file = os.path.join("../../keys/bonafide")

# Read keys

def load_keys(filepath):
    with open(filepath, 'r') as f:
        return set(os.path.basename(line.strip()) + '.npy' for line in f if line.strip())

# Load and sanitize keys
fake_files = load_keys(fake_key_file)
bonafide_files = load_keys(bonafide_key_file)



# Create destination directories
fake_dir = os.path.join("../spectrograms/ASVSpoof/fake")
working_dir = fake_dir
bonafide_dir = os.path.join("../spectrograms/ASVSpoof/bonafide")
os.makedirs(fake_dir, exist_ok=True)
os.makedirs(bonafide_dir, exist_ok=True)

# Move files
for filename in os.listdir(working_dir):
    src_path = os.path.join(working_dir, filename)
    print(f"src_path: {src_path}")
    print(f"filename: {filename}")

    print(f"Loaded {len(fake_files)} fake keys")
    print(f"Loaded {len(bonafide_files)} bonafide keys")
    # Skip directories and key files themselves
    if os.path.isdir(src_path) or filename in ['fake', 'bonafide']:
        continue

    if filename in fake_files:
        shutil.move(src_path, os.path.join(fake_dir, filename))
    elif filename in bonafide_files:
        shutil.move(src_path, os.path.join(bonafide_dir, filename))
    else:
        print(f"Skipping unknown file: {filename}")
