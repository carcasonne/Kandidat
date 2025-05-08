import os
from Datasets import ASVspoofDataset
from Visualizers import class_balance, avg_audio_length, hist_audio_length

def main():
    data_dir = "/home/alsk/Kandidat/AST/spectrograms/"
    output_dir = "/home/vson/JobOutputs/ASVspoofAnalysis/Output/"
    dataset = ASVspoofDataset.ASVspoofDataset(data_dir)

    # Run visualizations
    class_balance.plot_class_balance(dataset, output_dir)
    avg_audio_length.plot_avg_audio_length(dataset, output_dir)
    hist_audio_length.plot_hist_audio_length(dataset, output_dir)

if __name__ == "__main__":
    main()
