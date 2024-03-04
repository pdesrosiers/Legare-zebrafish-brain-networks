import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from random import uniform
from mpl_toolkits.axes_grid1 import make_axes_locatable
from calimba.analysis.utilities import *
import numpy as np
from scipy.stats import zscore
import sys
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches
from scipy.ndimage import gaussian_filter
from scipy import optimize
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from main import *


def rgb2hex(rgb_color):
    """Convert RGB color values to HEX code."""
    if rgb_color.dtype == 'float' and np.sum(rgb_color <= 1) == 3:
        rgb_color = (rgb_color * 255).astype('int')
    r, g, b = rgb_color
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def get_colors_cmap(cmap, n_colors, start=0, end=1, hex=False):
    sampling_points = np.linspace(start, end, n_colors, endpoint=True).astype('float')
    colors = []
    for s in sampling_points:
        rgb_color = plt.get_cmap(cmap)(s)[:3]
        if hex:
            hex_color = rgb2hex((np.array(rgb_color) * 255).astype('int'))
            colors.append(hex_color)
        else:
            colors.append(rgb_color)
    return colors


def merge(images, colors):
    """
    Merge a list of grayscale images (2D numpy arrays) into a single RGB image with normalization.

    Parameters:
    - images: List of 2D numpy arrays, each representing a grayscale image.
    - colors: List of tuples, each representing the RGB color to assign to the corresponding grayscale image.

    Returns:
    - A 3D numpy array representing the normalized merged RGB image.
    """
    # Ensure there's a color for each image
    if len(images) != len(colors):
        raise ValueError("Each image must have a corresponding color.")

    # Initialize the RGB image
    height, width = images[0].shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    # Merge each grayscale image into the RGB channels with normalization
    for img, color in zip(images, colors):
        for c in range(3):  # Iterate over RGB channels
            # Normalize based on the maximum intensity in each image to prevent saturation
            normalized_img = img / np.max(img) if np.max(img) > 0 else img
            rgb_image[:, :, c] += normalized_img * color[c]

    # Normalize the final image to be within [0, 1] range
    rgb_image /= np.max(rgb_image)

    return rgb_image


def hex2rgb(hex_color):
    h = hex_color.lstrip('#')
    return np.array(tuple(int(h[i:i+2], 16) for i in (0, 2, 4))) / 255


def barchart(ax, datasets, colors, delta=0.15, s=75, dotcolor='white', edgecolor='black', dot_alpha=0.5):
    for i, dataset in enumerate(datasets):
        ax.bar(i, np.mean(dataset), color=colors[i])
        ax.plot([i, i], [np.mean(dataset), np.mean(dataset) + np.std(dataset)], color='black')
        ax.plot([i - delta, i + delta], [np.mean(dataset) + np.std(dataset), np.mean(dataset) + np.std(dataset)], color='black')
        ax.scatter(np.random.normal(i, 0.05, len(dataset)), dataset, s=s, color=dotcolor, edgecolor=edgecolor, zorder=10, alpha=dot_alpha)


def plot_barchart_regions(ax, data_mean, data_std, labels, color='black', order=False, delta=0.1):
    if order:
        order = np.flip(np.argsort(data_mean))
    else:
        order = np.arange(0, len(labels))
    data_mean, data_std = np.copy(data_mean)[order], np.copy(data_std)[order]
    ax.bar(np.arange(0, len(labels)), data_mean, color=color)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks(np.arange(0, len(labels)), np.array(labels)[order], rotation=90, fontsize=12)
    for i in range(len(labels)):
        ax.plot([i, i], [data_mean[i], data_mean[i] + data_std[i]], color='black')
        ax.plot([i - delta, i + delta], [data_mean[i] + data_std[i], data_mean[i] + data_std[i]], color='black')


def line_plot(ax, datasets, labels, color='black'):
    for i in range(len(datasets[0])):
        ax.plot([0, 1], [datasets[0][i], datasets[1][i]], color=color)
        ax.scatter(0, datasets[0][i], s=50, facecolor=color, zorder=10)
        ax.scatter(1, datasets[1][i], s=50, facecolor=color, zorder=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_xlim([-0.5, 1.5])
    for position in ['top', 'right']:
        ax.spines[position].set_visible(False)


def compute_centroid_density(x, y, sigma=5, array_shape=(1000, 1000)):
    x_norm = x - np.min(x)
    x_norm = x_norm / np.max(x_norm)
    y_norm = y - np.min(y)
    y_norm = y_norm / np.max(y_norm)
    density = np.zeros(array_shape)
    for i, j in zip(x_norm, y_norm):
        density[int(j * array_shape[0] - 1), int(i * array_shape[1] - 1)] += 1
    density = gaussian_filter(density, sigma=sigma)
    density_values = []
    for i, j in zip(x_norm, y_norm):
        density_values.append(density[int(j * array_shape[0] - 1), int(i * array_shape[1] - 1)])
    return np.array(density_values)


def scatterplot_with_histograms(x, y, colors, labels):
    """
    Display a scatter plot with histograms for the x and y distributions along the top and right border of the frame,
    respectively.
    """
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    ax_histx.axis('off')
    ax_histy.axis('off')

    for i in range(len(x)):
        ax_scatter.scatter(x[i], y[i], color=colors[i], label=labels[i])
        ax_histx.hist(x, bins=50, color=colors[i])
        ax_histy.hist(y, bins=50, orientation='horizontal', color=colors[i])
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    ax_scatter.legend(fontsize=20)


def show(ax, matrix, **kwargs):
    image = ax.imshow(matrix, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(image, cax=cax)


def color_gradient(color1, color2, n_colors):
    colors = []
    for i in range(n_colors):
        alpha = i / (n_colors - 1)
        colors.append((1 - alpha) * np.array(color1) + alpha * np.array(color2))
    return colors


def generate_n_colors(nColors, saturation=80, value=90, randomness=0):
    h = np.linspace(0, 320, nColors)
    s = np.array([saturation + uniform(-randomness, randomness)] * nColors)
    v = np.array([value] * nColors)
    palette = []
    for i in range(nColors):
        palette.append(hsv_to_rgb(h[i] / 360, s[i] / 100, v[i] / 100))
    return palette


def set_boxplot_color(bp, color, linewidth=2, facecolor='none', markersize=5, marker='o'):
    for whisker in bp['whiskers']:
        whisker.set(color=color, linewidth=linewidth)
    for cap in bp['caps']:
        cap.set(color=color, linewidth=linewidth)
    for flier in bp['fliers']:
        flier.set(marker='.', color=color, alpha=1)
    for median in bp['medians']:
        median.set(color=color, linewidth=linewidth)
    for box in bp['boxes']:
        box.set(color=color, linewidth=linewidth)
        box.set(facecolor=facecolor)
    for flier in bp['fliers']:
        flier.set_marker(marker)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
        flier.set_markersize(markersize)


def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap


def display_half_matrices(ax, matrix1, matrix2, text1='Structural', text2='Functional', names=None, **kwargs):
    R = matrix1.shape[0]
    upperTriangle, lowerTriangle = triangle_indices(matrix1)
    composite = normalize(np.copy(matrix1))
    composite[upperTriangle] = normalize(matrix2)[upperTriangle]
    ax.imshow(composite, **kwargs)
    if names is not None:
        for i in range(R):
            ax.text(-1, i, names[i], horizontalalignment='right')
    ax.arrow(61.1, 61, -10, 0, width=0.5, head_width=1, color=[0, 0, 0])
    ax.arrow(61, 61.1, 0, -10, width=0.5, head_width=1, color=[0, 0, 0])
    ax.text(48, 61.6, text2, fontsize=20, horizontalalignment='right')
    ax.text(59.7, 48, text1, fontsize=20, rotation=90, verticalalignment='bottom')


def ridgeline(ax, timeseries, scale_factor=1, text=None, fontsize=16):
    L = timeseries.shape[0]
    x = np.linspace(0, timeseries.shape[1] - 1, timeseries.shape[1]).astype('int')
    for i in range(L):
        ax.plot(zscore(timeseries[i, :]) + scale_factor * (L - i), color=[0, 0, 0], zorder=i)
        plt.fill_between(x, zscore(timeseries[i, :]) + scale_factor * (L - i), 0, color=[1, 1, 1], zorder=i)
        if text is not None:
            ax.text(-5, scale_factor * (L - i), text[i], fontsize=fontsize, horizontalalignment='right')


def plot_clusters(ax, array, clusters, colors=None, aspect=2, cmap='hot', vmin=-1, vmax=7, interp='none', line_alpha=1):
    sorted_array = array[clusters == 0, :]
    sorted_clusters = clusters[np.argsort(clusters)]
    for c in np.unique(clusters)[1:]:
        sorted_array = np.append(sorted_array, array[clusters == c, :], axis=0)
    ax.imshow(sorted_array, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp)
    ax.plot([0, array.shape[1]], [0, 0], color='white')
    y = 0
    for i, c in enumerate(np.unique(sorted_clusters)):
        N = np.sum(clusters == c)
        if colors is not None:
            rect = patches.Rectangle((array.shape[1], y), 0.1 * array.shape[1], N, facecolor=colors[i], edgecolor='white')
            ax.add_patch(rect)
        y += N
        ax.plot([0, array.shape[1]], [y, y], color='white', alpha=line_alpha)
    if colors is not None:
        ax.set_xlim([0, 1.1 * array.shape[1]])
    else:
        ax.set_xlim([0, array.shape[1]])
    ax.set_ylim([array.shape[0], 0])


def plot_temporal_components(ax, components, variance):
    for i in range(components.shape[0]):
        ax.plot(normalize(components[i]) - i, color='black')
        ax.text(1.01 * components.shape[1], np.mean(normalize(components[i]) - i), '{:.3f}'.format(variance[i]))
    ax.axis('off')


def per_region_quantification(ax, values, region_labels, region_names, sort=False, color='black', highlight=-1):
    n_regions = len(region_names)
    mean_values = []
    for i in range(n_regions):
        if any(region_labels[:, i]):
            mean_values.append(np.mean(values[region_labels[:, i]]))
        else:
            mean_values.append(0)
    x = np.arange(n_regions) + 0.5
    if sort:
        order = np.flip(np.argsort(mean_values))
        ax.bar(x, np.array(mean_values)[order], facecolor=color)
        ax.set_xticks(x)
        ax.set_xticklabels(list(np.array(region_names)[order]), rotation=90, fontsize=10)
        if highlight >= 0:
            ax.bar(order[highlight] + 0.5, mean_values[order[highlight]], color='red')
    else:
        ax.bar(x, mean_values, facecolor=color)
        ax.set_xticks(x)
        ax.set_xticklabels(region_names, rotation=90, fontsize=10)
        if highlight >= 0:
            ax.bar(highlight + 0.5, mean_values[highlight], color='red')
    ax.plot([0, n_regions], [0, 0], color='black', linestyle='--')
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines['bottom'].set_bounds(0, n_regions)


class PaperFigure:

    def __init__(self, figsize=(7, 7), dpi=600):
        """
        Class to make multi-panel figures in Nature style. Notebooks which use this class should run this import
        statement before:
        from IPython.display import display

        Parameters
        ----------
        figsize: Size of the figure (width followed by height, in inches). Recommended width: 7.
        dpi: Resolution of the figure.
        """
        plt.ioff()
        plt.close('all')
        plt.ioff()
        self.set_tick_length()
        self.set_tick_pad()
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ratio = figsize[1] / figsize[0]  # Height / Width
        self.dims = figsize
        self.grid_dims = (1000, int(1000 * self.ratio))
        self.xscale = self.grid_dims[0] / figsize[0]
        self.yscale = self.grid_dims[1] / figsize[1]
        self.gridspec = gridspec.GridSpec(self.grid_dims[1], self.grid_dims[0], figure=self.fig)
        self.axes = {}

    @property
    def keys(self):
        keys = list(self.axes.keys())
        if 'background' in keys:
            keys.remove('background')
        return keys

    def set_font_size(self, fontsize):
        plt.rcParams['font.size'] = fontsize

    def add_axes(self, label, position, width, height):
        """label: 1 character string (a, b, c, ...) to label the panel
           position: top left corner coordinates (fractional, between 0 and 1)
           width: width of the panel (fractional, between 0 and 1)
           height: height of the panel (fractional, between 0 and 1)"""
        x0, y0 = int(position[0] * self.xscale), int(position[1] * self.yscale)
        x1, y1 = x0 + int(width * self.xscale), y0 + int(height * self.yscale)
        self.axes[label] = self.fig.add_subplot(self.gridspec[y0:y1, x0:x1])

    def add_background(self):
        """Positions empty array as a white background. Useful when first figuring out the layout to get full figure output,
        but should be removed when saving the figure."""
        self.add_axes('background', (0, 0), self.dims[0], self.dims[1])
        self.axes['background'].axis('off')

    def remove_all_ticks(self):
        for key in self.keys:
            ax = self.axes[key]
            ax.set_xticks([])
            ax.set_yticks([])

    def set_tick_length(self, length=1.5):
        plt.rcParams['xtick.major.size'] = length
        plt.rcParams['xtick.minor.size'] = length
        plt.rcParams['ytick.major.size'] = length
        plt.rcParams['ytick.minor.size'] = length

    def set_tick_pad(self, pad=3):
        plt.rcParams['xtick.major.pad'] = pad
        plt.rcParams['ytick.major.pad'] = pad

    def add_labels(self, ha='right', va='top', fontsize=14, weight='bold', padx=0.03, pady=0.01, labels=None):

        if labels is None:
            keys = self.keys
        else:
            keys = labels

        for key in keys:
            ax = self.axes[key]
            bbox = ax.get_position()
            x, y = bbox.x0, bbox.y1
            if len(key) == 1:
                self.fig.text(x - padx, y + pady, key, fontsize=fontsize, weight=weight, ha=ha, va=va)

    def show(self):
        display(self.fig)

    def save(self, path, pad=0):
        self.fig.savefig(path, bbox_inches='tight', pad_inches=pad)

    def close(self):
        self.fig.close()

    @staticmethod
    def get_cmap_color(cmap, value):
        """cmap: colormap string, value: number in range [0, 1]"""
        cmap = plt.cm.get_cmap(cmap)
        middle_color = cmap(value)
        color = mpl.colors.to_rgb(middle_color)
        return color

def plot_node_values(ax, values, atlas, excluded=None, view='top', double_vector=False, s=25, linewidth=1, cmap='hot', edgecolor='black'):
    if excluded is None:
        excluded = []
    if view == 'top':
        ax.imshow(atlas.XYprojection, cmap='gray')
        centroids = np.concatenate([atlas.regionCentroids['left'], atlas.regionCentroids['right']], axis=0)
        centroids[:, 1] = 974 - centroids[:, 1]
        if double_vector:
            values = double(values)
            excluded = np.concatenate([excluded, excluded + 70])
        centroids = np.delete(centroids, excluded, axis=0)
        ax.scatter(centroids[:, 0], centroids[:, 1], cmap=cmap, c=values, linewidth=linewidth, s=s, edgecolor=edgecolor)

"""
OLD DEPRECATED FUNCTION

def plot_network(ax, atlas, adjacency_matrix, excluded_regions=None, colors=cm.hot(range(256)), true_order=True, view='top', vmin=None, vmax=None, percentile=90, alpha=True):
    W = adjacency_matrix
    N_colors = colors.shape[0]
    if true_order:
        order = (((W - np.min(W)) / np.max(W - np.min(W))) * (N_colors - 1)).astype('int')
    else:
        order = (((W - np.percentile(W, percentile)) / np.max(W - np.percentile(W, percentile))) * (N_colors - 1)).astype('int')
        order[order < 0] = 0
    centroids = np.copy(atlas.regionCentroids['left'])
    centroids = np.append(centroids, atlas.regionCentroids['right'], axis=0)
    if excluded_regions is not None:
        centroids = np.delete(centroids, excluded_regions, axis=0)
    if view == 'top':
        image = atlas.XYprojection
        ax.imshow(image, cmap='gray')
        ax.scatter(centroids[:, 1], centroids[:, 0], color='white', s=100, edgecolor='black', zorder=10)
        for i in range(W.shape[0]):
            for j in range(i + 1, W.shape[1]):
                if (np.abs(W[i, j]) >= np.percentile(np.abs(W), percentile)) & (W[i, j] != 0):
                    if alpha:
                        alpha=(order[i, j]/(np.max(order))) ** 2
                    else:
                        alpha=1
                    ax.plot([centroids[i, 1], centroids[j, 1]],
                           [centroids[i, 0], centroids[j, 0]],
                            color=colors[order[i, j]],
                            alpha=alpha,
                            linewidth=1 + 5 * (order[i, j] / np.max(order)),
                            zorder=1 + (order[i, j] / np.max(order))
                           )
    elif view == 'side':
        image = atlas.XZprojection
        ax.imshow(image, cmap='gray')
        ax.scatter(centroids[:, 1], centroids[:, 2], color='white', s=100, edgecolor='black', zorder=10)
        for i in range(W.shape[0]):
            for j in range(i + 1, W.shape[1]):
                if (np.abs(W[i, j]) >= np.percentile(np.abs(W), percentile)) & (W[i, j] != 0):
                    if alpha:
                        alpha=(order[i, j]/(np.max(order))) ** 2
                    else:
                        alpha=1
                    ax.plot([centroids[i, 1], centroids[j, 1]],
                           [centroids[i, 2], centroids[j, 2]],
                            color=colors[order[i, j]],
                            alpha=alpha,
                            linewidth=1 + 3 * (order[i, j] / np.max(order)),
                            zorder=1 + (order[i, j] / np.max(order))
                           )
"""


def optimize_node_placement(centroids, graphWidth=597, maxiter=4, delta=25):
    """
    Expands the space between graph nodes to optimize edge display, without destroying global structure of the network.
    Achieved by minimizing an energy function with terms of repulsive energy between nodes and attractive energy with
    towards the center position of all nodes. Mirrors left nodes to the right hemisphere for symmetry. The gamma factor
    is used to set the attractive and repulsive terms in the same initial scale.
    Parameters
    ----------
    centroids : dict
        Dictionary with keys ['left', 'right'] which access the left and right hemisphere node centroids, respectively.
    centroids['left'] / centroids['right'] : numpy.ndarray
        Nx2 matrix with columns of X and Y coordinates for the N nodes of a brain hemisphere.
    graphWidth : int
        Heuristic parameter which sets the distance from which the left nodes are mirrored. Typically represents the
        pixel width of the image upon which the original graph would be projected.
    maxiter : int
        Number of optimize.minimize iterations. The larger the value, the more the nodes are spread apart. Could be
        limited by the delta parameter.
    delta : int
        Half-width of horizontal and vertical boxes which limit node movements. Larger values allow nodes to spread
        further apart.
    Returns
    -------
    nodesToDict(X, Y) : dict
        Dictionary in the same format as the input argument "centroids", with new optimized node positions.
    """

    X, Y = centroids[:, 0], centroids[:, 1]
    center = [np.mean(X), np.mean(Y)]
    X -= center[0]
    Y -= center[1]
    repulsive, attractive = computeInitialEnergyComponents(X, Y)
    gamma = attractive / repulsive
    initialGuess = np.array([X, Y]).ravel()
    bounds = []
    for position in initialGuess:
        bounds.append((position-delta, position+delta))
    results = optimize.minimize(computeEnergy, initialGuess, bounds=bounds, options={'maxiter': maxiter}, args=gamma)
    positions = results.x
    X = positions[:int(len(positions)/2)]
    Y = positions[int(len(positions)/2):]
    return X, Y


def computeEnergy(positions, gamma):
    """
    Function to optimize in the "optimizeNodePlacement" function. Measures the repulsive and attractive energy of nodes
    according to their positions.
    Parameters
    ----------
    positions : numpy.ndarray
        1-D array containing all X positions from all left and right hemisphere nodes, followed by all Y positions from
        the same nodes. Format [x_1, ... , x_N, y_1, ... , y_N]
    gamma : float
        Factor attributed to the repulsive energy term.
    Returns
    -------
    totalEnergy : float
        Total potential energy of the node positions.
    """
    X = positions[:int(len(positions)/2)]
    Y = positions[int(len(positions)/2):]
    N = len(X)
    distanceMatrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            distanceMatrix[i, j] = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            distanceMatrix[j, i] = distanceMatrix[i, j]
    totalEnergy = gamma * np.sum(1/distanceMatrix[np.triu_indices(N, 1)]) + np.sum(np.sqrt(X ** 2 + Y ** 2))
    return totalEnergy


def computeInitialEnergyComponents(X, Y):
    """
    Internal function used in the "optimizeNodePlacement" function. Computes the repulsive and attractive energy of
    the centered node distribution, which is then used to compute the gamma factor.
    Parameters
    ----------
    X : numpy.ndarray
        1-D array of all X coordinates of both left and right hemisphere nodes. Format [leftNodes --> rightNodes]
    Y : numpy.ndarray
        1-D array of all Y coordinates of both left and right hemisphere nodes. Format [leftNodes --> rightNodes]
    Returns
    -------
    repulsive : float
        Total repulsive energy of nodes (1/r potential)
    attractive : float
        Total attractive energy of nodes with regard to the origin (r potential)
    """
    N = len(X)
    distanceMatrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            distanceMatrix[i, j] = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            distanceMatrix[j, i] = distanceMatrix[i, j]
    repulsive = np.sum(1/distanceMatrix[np.triu_indices(N, 1)])
    attractive = np.sum(np.sqrt(X ** 2 + Y ** 2))
    return repulsive, attractive

