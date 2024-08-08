import torch

"""
models_paths = ['path/to/fold1_model.pth', 'path/to/fold2_model.pth', 'path/to/fold3_model.pth', 'path/to/fold4_model.pth', 'path/to/fold5_model.pth']
models = [load_model(path) for path in models_paths]
"""

def ensemble_predict(models, input):
    with torch.no_grad():
        logits_sum = 0
        for model in models:
            logits = model(input)
            logits_sum += logits
        logits_mean = logits_sum / len(models)
        probabilities = torch.softmax(logits_mean, dim=1)
    return probabilities
