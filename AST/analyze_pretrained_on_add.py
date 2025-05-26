import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from Datasets import ADDdatasetPretrain, StretchMelCropTime
import modules.models as models
import modules.analysis as analysis
import vson_statics as statics

# Load model
model = models.load_pretrained_model(
    saved_model_path=statics.PRETRAIN_MODEL_CHECKPOINT,
    device=statics.DEVICE
)

# Setup transform for pretrained model
transform = transforms.Compose([
    StretchMelCropTime(224, 224),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Load dataset
dataset = ADDdatasetPretrain(
    data_dir=statics.ADD_DATASET_PATH,
    max_per_class=statics.k_samples,
    transform=transform
)
loader = DataLoader(dataset, batch_size=statics.BATCH_SIZE, shuffle=False)

# Analyze with results caching
analyzer = analysis.quick_analysis(
    model=model,
    dataloader=loader,
    dataset_name="Pretrained_ADD",
    device=statics.DEVICE,
    is_ast=False,
    save_dir="results/pretrained_add",
    model_path=statics.PRETRAIN_MODEL_CHECKPOINT
)

print("Analysis complete! Check results/pretrained_add/ for outputs.")