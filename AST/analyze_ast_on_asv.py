import modules.models as models
import modules.analysis as analysis
import modules.dataset_cache as dataset_cache
import vson_statics as statics

# Load model
model = models.load_modified_ast_model(
   base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
   finetuned_model_path=statics.AST_MODEL_CHECKPOINT,
   device=statics.DEVICE
)

# Load dataset with caching
loader = dataset_cache.get_dataset(
   dataset_type="ADD",
   dataset_path=statics.ADD_DATASET_PATH,
   model_type="ast",
   max_per_class=statics.k_samples,
   batch_size=16,
   target_frames=statics.AST_TARGET_FRAMES
)

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