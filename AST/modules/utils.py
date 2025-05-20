def get_input_and_labels(is_AST, batch, device):
    if is_AST:
        inputs = batch["input_values"].to(device)  # shape: (B, T, 128)
        labels = batch["labels"].to(device)
        return inputs, labels
    else:
        inputs, labels = batch  # batch is a tuple (inputs, labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels