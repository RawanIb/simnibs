'''
2017 Zhu et al _ A 3D Computational Model of Transcutaneous Electrical Nerve Stimulation for Estimating Aβ Tactile Nerve Fiber Excitability.pdf
'''

import os
import time
from dataclasses import dataclass

import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from simnibs import sim_struct, run_simnibs


# Layer thicknesses from paper (in mm)
@dataclass
class LayerElement:
    thickness: float
    order: int

    @property
    def label(self):
        return self.order

    @property
    def thickness_voxel(self):
        return round(self.thickness / voxel_size) * voxel_size


class LayerTag:
    # BONE_MARROW = LayerElement(thickness=6.5, order=1)
    # CORTICAL_BONE = LayerElement(thickness=6, order=2)
    # MUSCLE = LayerElement(thickness=13.5, order=3)
    FAT = LayerElement(thickness=2.5, order=4)
    DERMIS = LayerElement(thickness=1.411, order=5)
    EPIDERMIS = LayerElement(thickness=0.06, order=6)
    STRATUM_CORNEUM = LayerElement(thickness=0.029, order=7)


# Get layers in order from innermost to outermost
layers = [layer for layer in LayerTag.__dict__.values() if isinstance(layer, LayerElement)]
layers.sort(key=lambda x: x.order)

radius = sum(layer.thickness for layer in layers)
print(f"Total radius: {radius:.2f}mm")

# Define physical dimensions in mm
full_length = 150  # total forearm length
slice_length = 1  # length of the high-resolution section to model
voxel_size = 0.01  # 10um micrometers to capture stratum corneum
# voxel_size = 0.05 # 0.5mm to capture all layers

# Calculate dimensions
x_size = int(slice_length / voxel_size)
y_size = int(2 * radius / voxel_size)
z_size = int(2 * radius / voxel_size)

print(f"Single slice dimensions: {x_size} x {y_size} x {z_size}")

# Create coordinate grids for one slice
yv, zv = np.meshgrid(
    np.linspace(-radius, radius, y_size),
    np.linspace(-radius, radius, z_size),
    indexing='ij'
)

# Calculate radius for each point
r = np.sqrt(yv ** 2 + zv ** 2)
plt.imshow(r)
plt.colorbar()
plt.title(f"Radius (mm), slice radius: {radius:.2f}mm")
plt.show()

# Create and verify single slice
slice_img = np.zeros((y_size, z_size), np.uint16)
current_radius = 0
for layer in layers:
    current_radius += layer.thickness_voxel
    mask = (r <= current_radius)
    slice_img[mask & (slice_img == 0)] = layer.label
plt.imshow(slice_img[200:300, :1000])
plt.colorbar()
plt.title("Layer labels")
plt.show()
print(max(r.flatten()))

# Verify layer creation
print("\nLayer verification:")
print("Layer | Thickness(mm) | Voxels | Actual(mm)")
print("-" * 50)
for layer in layers:
    voxel_count = np.sum(slice_img == layer.label)
    radial_voxels = np.max(np.sum(slice_img == layer.label, axis=1))
    actual_thickness = radial_voxels * voxel_size
    print(f"{layer.order:2d} | {layer.thickness:11.3f} | {radial_voxels:6d} | {actual_thickness:.3f}")

# Calculate how many slices we need for full length
n_slices = int(full_length / slice_length)
print(f"Number of slices needed: {n_slices}")

# Create the full volume by replicating the slice
tic_time = time.time()
full_img = np.zeros((n_slices * x_size, y_size, z_size), np.uint16)
for i in range(n_slices):
    start_idx = i * x_size
    end_idx = (i + 1) * x_size
    full_img[start_idx:end_idx] = slice_img
print(f"Created full volume in {time.time() - tic_time:.2f}s")

# Save the volume
affine = np.eye(4)
affine[0:3, 0:3] *= voxel_size  # Set voxel size in affine matrix
tic_time = time.time()
img = nib.Nifti1Image(full_img, affine)
nib.save(img, 'forearm_r4l150mm.nii.gz')
print(f"Saved full volume in {time.time() - tic_time:.2f}s")