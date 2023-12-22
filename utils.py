def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def print_metrics(metrics):
    return {k:v.item() for k, v in metrics.items()}