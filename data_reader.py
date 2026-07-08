import numpy as np
import matplotlib.pyplot as plt
import os

path = "./data/Train/HIGH/Train_000_HiGH.bin"

shape = (200, 200, 512)   # [H, W, D]
dtype = np.uint16
offset = 48

raw = np.fromfile(path, dtype=dtype, offset=offset)
volume = raw.reshape(shape)

print("Volume shape:", volume.shape)
print("Volume min:", volume.min())
print("Volume max:", volume.max())
print("Volume mean:", volume.mean())

# MAP / MIP image along depth axis
map_img = np.max(volume, axis=2)   # [H, W, D] -> [H, W]

print("MAP shape:", map_img.shape)
print("MAP min:", map_img.min())
print("MAP max:", map_img.max())
print("MAP mean:", map_img.mean())

# contrast adjustment
vmin, vmax = np.percentile(map_img, [0, 99])

plt.figure(figsize=(6, 6))
plt.imshow(map_img, cmap="hot", vmin=vmin, vmax=vmax)
plt.title("MAP image")
plt.axis("off")
plt.colorbar()
plt.show()