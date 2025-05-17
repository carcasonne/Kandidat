import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from ADD_train_benchmark import *
def load_modified_ast_model(base_model_name, finetuned_model_path, device=None):
    """
    Load a model where only the last two layers are replaced with fine-tuned weights.

    Args:
        base_model_name: Name of the original pretrained model to start with
        finetuned_model_path: Path to the saved model with fine-tuned weights
        device: The device to load the model to ('cuda', 'cpu', or None to auto-detect)

    Returns:
        Model with base weights plus fine-tuned last layers
    """
    # Determine device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Loading base model {base_model_name}")
    # Start with the original pretrained model
    model = ASTForAudioClassification.from_pretrained(base_model_name)

    # Apply architecture modifications
    model.config.max_length = 300
    model.config.num_labels = 2
    model.config.id2label = {0: "bonafide", 1: "fake"}
    model.config.label2id = {"bonafide": 0, "fake": 1}

    # Modify the classifier for 2 classes (same as in your training code)
    model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)
    model.classifier.out_proj = nn.Linear(2, 2)

    # Interpolate positional embeddings
    desired_max_length = 350
    position_embeddings = model.audio_spectrogram_transformer.embeddings.position_embeddings
    old_len = position_embeddings.shape[1]
    if old_len != desired_max_length:
        print(f"Interpolating position embeddings from {old_len} to {desired_max_length}")
        interpolated_pos_emb = F.interpolate(
            position_embeddings.permute(0, 2, 1),
            size=desired_max_length,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)
        model.audio_spectrogram_transformer.embeddings.position_embeddings = nn.Parameter(interpolated_pos_emb)

    # Load state dict from the fine-tuned model using safetensors format
    print(f"Loading fine-tuned weights from {finetuned_model_path}")

    try:
        # Try to load using safetensors
        from safetensors import safe_open
        from safetensors.torch import load_file

        safetensors_path = os.path.join(finetuned_model_path, "model.safetensors")
        print(f"safetensors path: {safetensors_path}")
        if os.path.exists(safetensors_path):
            print(f"Loading model from safetensors file: {safetensors_path}")
            finetuned_state_dict = load_file(safetensors_path)
        else:
            # Fall back to regular pytorch model loading
            pytorch_path = os.path.join(finetuned_model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_path):
                print(f"Loading model from PyTorch file: {pytorch_path}")
                finetuned_state_dict = torch.load(pytorch_path, map_location=device)
            else:
                # If neither file exists, try loading directly from the path
                print(f"Attempting to load model directly from: {finetuned_model_path}")
                model = ASTForAudioClassification.from_pretrained(finetuned_model_path, local_files_only=True)
                # Re-freeze the layers after loading
                N = 10
                for i in range(N):  # Layers 0 to 9
                    for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
                        param.requires_grad = False
                model = model.to(device)
                print(f"Model successfully loaded directly")
                return model

    except (ImportError, FileNotFoundError) as e:
        print(f"Error loading model: {e}")
        print("Trying alternate loading method...")

        # Try loading the model directly using from_pretrained
        try:
            print(f"Loading fine-tuned model directly from {finetuned_model_path}")
            model = ASTForAudioClassification.from_pretrained(finetuned_model_path, local_files_only=True)

            # Re-freeze the layers after loading
            N = 10
            for i in range(N):  # Layers 0 to 9
                for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
                    param.requires_grad = False

            model = model.to(device)
            print(f"Model successfully loaded directly")
            return model
        except Exception as e2:
            print(f"Failed to load model directly: {e2}")
            raise e2

    # If we got here, we have a state dict to filter
    # Filter the state dict to only include the last two transformer layers and classifier
    last_layers_dict = {}
    for key, value in finetuned_state_dict.items():
        # Include only the last two transformer layers (layers 10 and 11)
        if "encoder.layer.10." in key or "encoder.layer.11." in key:
            last_layers_dict[key] = value
        # Include classifier weights
        elif "classifier" in key:
            last_layers_dict[key] = value

    print(f"Selectively loading {len(last_layers_dict)} weights for the last layers and classifier")

    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(finetuned_state_dict, strict=True)
    print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    # Move the model to the specified device
    model = model.to(device)
    print("Model successfully prepared with fine-tuned last layers")

    return model

def extract_embeddings_from_multiple_dataloaders(model, dataloaders, device='cuda'):
    """
    Extract [CLS] token embeddings from AST model across multiple dataloaders,
    assigning domain IDs (0, 1, 2) to each dataset.

    Args:
        model: your AST model
        dataloaders: list of 3 PyTorch DataLoaders
        device: 'cuda' or 'cpu'

    Returns:
        embeddings: Tensor (N, hidden_dim)
        labels: Tensor (N,)
        domains: Tensor (N,)
    """
    model.eval()
    embeddings = []
    labels = []
    domains = []

    def hook_fn(module, input, output):
        # Grab [CLS] token embedding
        hook_fn.embeddings = output[0][:, 0, :].detach()

    handle = model.audio_spectrogram_transformer.encoder.register_forward_hook(hook_fn)

    with torch.no_grad():
        for domain_id, dataloader in enumerate(dataloaders):
            print(f" Domain_ID: {domain_id}")
            for batch in dataloader:
                input_values = batch['input_values'].to(device)
                label = batch['labels'].to(device)

                _ = model(input_values)
                emb = hook_fn.embeddings

                embeddings.append(emb.cpu())
                labels.append(label.cpu())
                domains.append(torch.full_like(label, fill_value=domain_id, dtype=torch.long))

    handle.remove()

    return torch.cat(embeddings), torch.cat(labels), torch.cat(domains)


def visualize(embeddings, labels, domains):

    reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    reduced = reducer.fit_transform(embeddings.numpy())

    # Convert to DataFrame for easy plotting
    import pandas as pd
    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": labels.numpy(),
        "domain": domains.cpu().numpy()
    })
    df = df.sample(n=min(len(df), 2000), random_state=42)
    # Plot by spoof/real
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="coolwarm")
    plt.title("Colored by Spoof/Real")

    # Plot by dataset
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x="x", y="y", hue="domain", palette="tab10")
    plt.title("Colored by Dataset")

    plt.tight_layout()
    plt.savefig("embedding_visualization.png")

print("Inside script")
MODEL_CHECKPOINT = "checkpoints/asvspoof-ast-model15_100K_20250506_054106"

print("Loading model")
model = load_modified_ast_model(
    base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",  # Original model name
    finetuned_model_path=MODEL_CHECKPOINT,      # Your saved model
    device="cuda"
)
print("Done loading model")
vson = True
if vson:
    ADD_DATASET_PATH = r"/home/alsk/Kandidat/AST/spectrograms/ADD"
    FOR_DATASET_PATH = r"/home/alsk/Kandidat/AST/spectrograms/FoR/for-2sec/for-2seconds"
    FOR_DATASET_PATH_TRAINING = r"/home/alsk/Kandidat/AST/spectrograms/FoR/for-2sec/for-2seconds/Training"
    FOR_DATASET_PATH_TESTING = r"/home/alsk/Kandidat/AST/spectrograms/FoR/for-2sec/for-2seconds/Testing"
    ASVS_DATASET_PATH = r"/home/alsk/Kandidat/AST/spectrograms"
else:
    # Define dataset path
    ADD_DATASET_PATH = r"spectrograms/ADD"
    FOR_DATASET_PATH = r"spectrograms/FoR/for-2sec/for-2seconds"
    FOR_DATASET_PATH_TRAINING = r"spectrograms/FoR/for-2sec/for-2seconds/Training"
    FOR_DATASET_PATH_TESTING = r"spectrograms/FoR/for-2sec/for-2seconds/Testing"
    ASVS_DATASET_PATH = r"spectrograms"

print("Loading asv dataset")
asv_data, _, _ = load_ASV_dataset(ASVS_DATASET_PATH, samples_asv, is_AST=True, split=None, transform=None, embedding_size=300)
print("Loading ADD dataset")
add_data, _, _ = load_ADD_dataset(ADD_DATASET_PATH, samples_add, True, TRAIN_TEST_SPLIT, embedding_size=300)
print("Loading FoR dataset")
for_data, _, _ = load_FOR_dataset(FOR_DATASET_PATH_TRAINING, FOR_DATASET_PATH_TESTING, True, samples_for, None, 300)

print("Extracting embeddings")
embeddings, labels, domains = extract_embeddings_from_multiple_dataloaders(model, [asv_data, add_data, for_data], device='cuda')
visualize(embeddings, labels, domains)