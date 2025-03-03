import torch

def merge_models(model_a, model_b, alpha=0.5):
    """
    Averages the weights of two models using a given alpha.
    
    Args:
        model_a (dict): State dict of model A.
        model_b (dict): State dict of model B.
        alpha (float): Interpolation factor (0.0 = all B, 1.0 = all A).

    Returns:
        dict: Merged model state dict.
    """
    merged_model = {}
    for key in model_a.keys():
        if key in model_b:
            merged_model[key] = alpha * model_a[key].to(torch.float32) + (1 - alpha) * model_b[key].to(torch.float32)
        else:
            merged_model[key] = model_a[key]
    return merged_model

if __name__ == "__main__":
    model_a_path = "path_to_model_a.pth"
    model_b_path = "path_to_model_b.pth"
    output_path = "merged_model.pth"
    alpha = 0.5

    model_a = torch.load(model_a_path, map_location="cpu")
    model_b = torch.load(model_b_path, map_location="cpu")

    merged_model = merge_models(model_a, model_b, alpha)
    
    # Convert back to FP16 where applicable
    for key in merged_model.keys():
        if model_a[key].dtype == torch.float16 or model_b[key].dtype == torch.float16:
            merged_model[key] = merged_model[key].to(torch.float16)
    
    torch.save(merged_model, output_path)
    print(f"Merged model saved to {output_path}")
