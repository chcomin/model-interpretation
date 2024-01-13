import numpy as np
import matplotlib.pyplot as plt

def create_image(model, data, eps=0.1, grid_resolution=300):
    """Create an image containing the output of a model for each point 
    of a grid. Used for visualizing the decision surface of a model."""

    x0, x1 = data.T

    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )

    data_grid = np.c_[xx0.ravel(), xx1.ravel()]

    response = model(data_grid)
    response = response.reshape(xx0.shape)

    return response, xx0, xx1

def plot_regions(model, data, labels, grid_resolution=300, eps=0.1):
    """Plot the output of a model for a dense grid of points bounded by the data."""

    #colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
    #      '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']

    response, xx0, xx1 = create_image(model, data, eps=eps, grid_resolution=grid_resolution)

    fig, ax = plt.subplots(figsize=(7,6))
    co = ax.pcolormesh(xx0, xx1, response, 
                    cmap='tab10', alpha=0.5)
    sc = ax.scatter(*data.T, s=3, c=labels, cmap='tab10')
    fig.colorbar(sc, ax=ax)
