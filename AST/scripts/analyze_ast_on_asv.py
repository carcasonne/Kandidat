from torch.utils.data import DataLoader
from Datasets import ADDdataset
import modules.models as models
import modules.analysis as analysis
import scripts.vson_statics as statics

# Quick configuration
# Load model and data
model = models.load_modified_ast_model(
    base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
    finetuned_model_path=statics.AST_MODEL_PATH,
    device=statics.DEVICE
)

dataset = ADDdataset(statics.ADD_DATA_PATH, max_per_class=statics.k_samples, target_frames=statics.AST_TARGET_FRAMES)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Analyze
analyzer = analysis.quick_analysis(
    model=model,
    dataloader=loader,
    dataset_name="AST_ADD",
    device=statics.DEVICE,
    is_ast=True,
    save_dir="results/ast_add"
)

print("Analysis complete! Check results/ast_add/ for outputs.")
