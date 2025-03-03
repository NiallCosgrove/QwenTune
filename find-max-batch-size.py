import torch

def find_max_batch_size(model, input_shape, device="cuda"):
    batch_size = 8  # Start conservative
    while True:
        try:
            dummy_input = torch.randint(0, 50257, (batch_size, input_shape), device=device)
            _ = model(dummy_input)
            print(f"Batch size {batch_size} ✅ fits")
            batch_size += 2  # Increment cautiously
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} ❌ OOM")
                return batch_size - 2  # Back off to last working size
            else:
                raise  # Other errors should be surfaced

# Example usage
# model = load_your_model()  # Load your model
# max_batch = find_max_batch_size(model, 1119)
# print(f"Optimal batch size: {max_batch}")
