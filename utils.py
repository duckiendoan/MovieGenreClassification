from PIL import Image
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def pretty_metrics(metrics):
    return {k:v.item() for k, v in metrics.items()}

def compute_mean_std(img_paths):
  mean = np.array([0., 0., 0.])
  for path in img_paths:
    img = Image.open(path)
    if img.mode != 'RGB':
      img = img.convert('RGB')
    A = np.asarray(img).astype(float) / 255.
    for c in range(3):
      mean[c] += A[:, :, c].mean()
  mean = mean / len(img_paths)
  var = np.zeros_like(mean)
  for path in img_paths:
    img = Image.open(path)
    if img.mode != 'RGB':
      img = img.convert('RGB')
    A = np.asarray(img).astype(float) / 255.
    for c in range(3):
      var[c] += ((A[:, :, c] - mean[c]) ** 2).mean()
  var = var / len(img_paths)
  return mean, np.sqrt(var)