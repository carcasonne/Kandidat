# Kandidat

This GitHub repository contains the python scripts used to load, run, and evaluate the models for our Master Thesis, which can be found in the 'AST' folder.
It also contains a project for creating consistent figures from the weights and biases project logs in the DataVisualization folder.

The code is setup in such a way that you should write your needed scripts yourself in the AST folder, utilizing the existing classes to load the needed datasets and models, and train/benchmark your model on the chosen data. Look at AST/train.py for inspiration on how to do it.

# Datasets

The system is set up to look for pre-generated spectrogram (in .npy format).The folder structure we used for the three expected datasets: ASV21, ADD, FoR-2sec:
```
spectrograms/
├── ADD/
│   ├── fake/
│   └── genuine/
├── ASVSpoof/
│   ├── bonafide/
│   └── fake/
├── FoR/
│   └── for-2sec/
│       └── for-2seconds/
│           ├── Testing/
|               └── Fake/
|               └── Real/
│           └── Training/
|               └── Fake/
|               └── Real/
```
