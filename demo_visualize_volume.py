import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if False:
    # Generate a small volume (e.g., 20x20x20)
    x, y, z = np.indices((20, 20, 20))
    volume = np.exp(-(x-10)**2/20 - (y-10)**2/20 - (z-10)**2/20)  # Gaussian blob

    # Flatten and filter low-intensity voxels
    mask = volume > 0.01  # Threshold to reduce points
    x_pts, y_pts, z_pts = x[mask], y[mask], z[mask]
    colors = 100*volume[mask]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_pts, y_pts, z_pts, c=colors, cmap='hot', alpha=0.1, s=100)
    plt.colorbar(scatter, label='Intensity')
    ax.set_title("Pseudo-Volume Rendering (3D Scatter)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


if False:
    # Generate a synthetic 3D volume (e.g., a sphere)
    x, y, z = np.mgrid[-10:10:30j, -10:10:30j, -10:10:30j]
    volume = np.sin(np.sqrt(10*x**2 + y**2 + z**2))  # Spherical gradient

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot slices with transparency
    for z_val in np.linspace(-10, 10, 20):
        z_idx = int((z_val + 10) / 20 * 29)  # Map to volume index
        slice_data = volume[:, :, z_idx]
        x_slice, y_slice = np.mgrid[-10:10:30j, -10:10:30j]
        ax.plot_surface(x_slice, y_slice, np.full_like(x_slice, z_val),
                        facecolors=plt.cm.Greys(slice_data),
                        rstride=1, cstride=1, alpha=0.01, shade=True)

    ax.set_title("Layered Slices-Volume Rendering ")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

# Generate a smooth volume (sphere + cube)
x, y, z = np.indices((30, 30, 30))
sphere = (x-15)**2 + (y-15)**2 + (z-15)**2 < 64  # Radius=8
cube = (x > 5) & (x < 10) & (y > 5) & (y < 10) & (z > 5) & (z < 10)
volume = np.logical_or(sphere, cube).astype(float)

# Extract surfaces
verts, faces, _, _ = marching_cubes(volume, level=0.5)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
#mesh = Poly3DCollection(verts[faces], alpha=0.3, edgecolor='b', facecolor='blue')
mesh = Poly3DCollection(verts[faces], alpha=0.3, facecolor='blue')
ax.add_collection3d(mesh)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 30)
ax.set_title("Smooth Shapes (Fixed Alpha)")
plt.tight_layout()
plt.show()