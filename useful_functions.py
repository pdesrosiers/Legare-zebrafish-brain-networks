from matplotlib.collections import LineCollection
from main import *
from visualization import *


def get_correlation_matrices(timeseries_list):
    corr_matrices = []
    for i in range(len(timeseries_list)):
        matrix = compute_correlation_matrix(timeseries_list[i])
        matrix[np.isnan(matrix)] = 0
        corr_matrices.append(matrix)
    return corr_matrices


def get_datasets(top_directory, keywords=[], exclude=[]):
    folders = identify_folders(top_directory, ['dpf'])
    datasets = []
    for folder in folders:
        datasets += identify_folders(folder, keywords=keywords, exclude=exclude)
    return datasets


def identify_files(path, keywords=None, exclude=None):
    items = os.listdir(path)
    if keywords is None:
        keywords = []
    if exclude is None:
        exclude = []
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            if any(excluded in item for excluded in exclude):
                pass
            else:
                files.append(item)
    files.sort()
    return files


def identify_folders(path, keywords=None, exclude=None):
    initial_folders = [f.path for f in os.scandir(path) if f.is_dir()]
    if keywords is None:
        keywords = []
    if exclude is None:
        exclude = []
    folders = []
    for folder in initial_folders:
        if all(keyword in folder for keyword in keywords):
            if any(excluded in folder for excluded in exclude):
                pass
            else:
                folders.append(folder)
    for i in range(len(folders)):
        folders[i] += '/'
    folders.sort()
    return folders
    

def get_edgelist(matrix, directed=False):
    if directed:
        i, j = np.where(matrix != 0)
        edges = np.stack([i, j], axis=1).astype('int')
    else:
        triangle = np.triu_indices(matrix.shape[0], 1)
        matrix_with_zeros = np.copy(matrix)
        matrix_with_zeros[triangle] = 0
        matrix__with_zeros = matrix_with_zeros.T
        i, j = np.where(matrix_with_zeros != 0)
        edges = np.stack([i, j], axis=1).astype('int')
    weights = matrix[i, j]
    edgelist = []
    for i, edge in enumerate(edges):
        edgelist.append(tuple([edge[0], edge[1], weights[i]]))
    return edgelist


def get_edge_positions(centroids, edgelist):
    x1, x2, y1, y2 = [], [], [], []
    for edge in edgelist:
        i, j, _ = edge
        x1.append(centroids[int(i), 0])
        y1.append(centroids[int(i), 1])
        x2.append(centroids[int(j), 0])
        y2.append(centroids[int(j), 1])
    edgepos = []
    for i in range(len(x1)):
        edgepos.append([(x1[i], y1[i]), (x2[i], y2[i])])
    return edgepos


def draw_graph(ax, adjacency, centroids, directed=False, percentile=75, s=40, node_color='white', linewidth=1.5, edge_color='white', alpha=0.75, edge_cmap='hot', edge_vmin=0, edge_vmax=1, flip_order=False, node_edgecolor='black', node_edgewidth=1):
    edge_list = get_edgelist(adjacency)
    edge_array = np.array(edge_list)
    edge_ids = edge_array[:, 2] >= np.percentile(edge_array[:, 2], percentile)
    edge_list = list(edge_array[edge_ids, :])
    edge_position = get_edge_positions(centroids, edge_list)
    weights = np.array(edge_list)[:, 2]
    order = np.argsort(weights)
    if flip_order:
        order = np.flip(order)
    sorted_positions = []
    for i in order:
        sorted_positions.append(edge_position[i])
    edge_collection = LineCollection(
            sorted_positions,
            array=weights[order],
            cmap=edge_cmap,
            linewidths=weights[order] * linewidth,
            antialiaseds=(1,),
            alpha=(np.abs(weights[order]) / np.max(np.abs(weights[order]))) * alpha
        )
    edge_collection.set_clim(edge_vmin, edge_vmax)
    edge_collection.set_zorder(1)
    ax.scatter(centroids[:, 0], centroids[:, 1], color=node_color, zorder=10, s=s, edgecolor=node_edgecolor, linewidths=node_edgewidth)
    ax.add_collection(edge_collection)
