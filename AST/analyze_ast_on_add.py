import torch
from torch.utils.data import DataLoader
from Datasets import ADDdataset
import modules.models as models
import modules.analysis as analysis
import vson_statics as statics

# Load model
model = models.load_modified_ast_model(
  base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
  finetuned_model_path=statics.AST_MODEL_CHECKPOINT,
  device=statics.DEVICE
)

# Load dataset
dataset = ADDdataset(
  data_dir=statics.ADD_DATASET_PATH,
  max_per_class=statics.all_samples,
  target_frames=statics.AST_TARGET_FRAMES
)
loader = DataLoader(dataset, batch_size=statics.BATCH_SIZE, shuffle=False)

# Analyze with results caching
analyzer = analysis.quick_analysis(
  model=model,
  dataloader=loader,
  dataset_name="AST_ADD",
  device=statics.DEVICE,
  is_ast=True,
  save_dir="results/ast_add",
  model_path=statics.AST_MODEL_CHECKPOINT
)

print("Analysis complete! Check results/ast_add/ for outputs.")