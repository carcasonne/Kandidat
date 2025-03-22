import os

def process_keys(in_path, out_path):
    bonafide_samples = []
    fake_samples = []
    os.makedirs(out_path, exist_ok=True)

    with open(in_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if line.strip():
                columns = line.split()
                # Check if the second-to-last column is 'bonafide'
                if "bonafide" in line:
                    bonafide_samples.append(columns[1])  # Append the sample name (second column)
                elif "spoof" in line:
                    fake_samples.append(columns[1])

    with open(os.path.join(out_path, "bonafide"), 'w') as file:
        for sample in bonafide_samples:
            if(os.path.exists(f"ASVSpoof/ASVspoof2021_DF_eval/flac/{sample}.flac")):
                file.write(sample + '\n')

    with open(os.path.join(out_path, "fake"), 'w') as file:
        for sample in fake_samples:
            if(os.path.exists(f"ASVSpoof/ASVspoof2021_DF_eval/flac/{sample}.flac")):
                file.write(sample + '\n')



process_keys("alldata.txt", "./")