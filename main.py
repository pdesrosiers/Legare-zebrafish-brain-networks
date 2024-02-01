from calimba.analysis.utilities import *
from skimage import io
from scipy.sparse import load_npz
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
from tqdm import tqdm
from scipy.signal import medfilt
from numba import njit
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
import h5py


def get_region_name(atlas, keyword):
    region_ids = []
    for i, name in enumerate(atlas.names):
        if keyword in name:
            region_ids.append(i)
    return region_ids


@njit
def compute_correlation_matrix(timeseries, arctanh=False):
    N = timeseries.shape[0]
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if arctanh:
                matrix[i, j] = np.arctanh(np.corrcoef(timeseries[i], timeseries[j])[0, 1])
            else:
                matrix[i, j] = np.corrcoef(timeseries[i], timeseries[j])[0, 1]
            matrix[j, i] = matrix[i, j]
    return matrix


def baseline_minfilter(signal, window=300, sigma1=5, sigma2=100, debug=False):
    signal_flatstart = np.copy(signal)
    signal_flatstart[0] = signal[1]
    smooth = gaussian_filter1d(signal_flatstart, sigma1)
    mins = minimum_filter1d(smooth, window)
    baseline = gaussian_filter1d(mins, sigma2)
    if debug:
        debug_out = np.asarray([smooth, mins, baseline])
        return debug_out
    else:
        return baseline


def compute_dff_using_minfilter(timeseries, window=200, sigma1=0.1, sigma2=50):
    dff = np.zeros(timeseries.shape)
    for i in range(timeseries.shape[0]):
        if np.any(timeseries[i]):
            baseline = baseline_minfilter(timeseries[i], window=window, sigma1=sigma1, sigma2=sigma2)
            dff[i] = (timeseries[i] - baseline) / baseline
    return dff


def load_metadata(directory):
    files = identify_files(directory, ['info', '.txt'])
    if any(files):
        if len(files) > 1:
            raise Warning('There is more than one info file. Loading the first one.')
        metadata = read_info_file(directory + files[0])
        return metadata
    else:
        raise FileNotFoundError('No info file found in directory.')


def read_info_file(path, strings=False):
    f = open(path, 'r')
    info = f.readlines()
    f.close()
    metadata = {}
    for line in info:
        attribute, value = line.split('=')[0].strip(), line.split('=')[1].strip()
        metadata[attribute] = value
    if strings is False:
        for key in metadata.keys():
            if key in ['laser.power', 'x.pixels', 'y.pixels', 'z.planes', 't.frames',
                             'frame.rate', 'volume.rate', 'x.pixel.size', 'y.pixel.size', 'z.spacing']:
                try:
                    metadata[key] = float(metadata[key])
                except:
                    pass
    return metadata


def load_data(directory):
    files = identify_files(directory, ['data', '.hdf5'])
    if any(files):
        if len(files) > 1:
            raise Warning('Multiple data files in directory. Loading the first one.')
        return load_hdf5(directory + files[0])
    else:
        raise FileNotFoundError('No .hdf5 data file identified in directory.')


def load_hdf5(path):
    data = {}
    file = h5py.File(path, 'r')
    for dataset in file.keys():
        data[dataset] = np.array(file[dataset])
    file.close()
    return data


class MapzebrainAtlas:

    path = '/home/tonepone/Documents/Registration/StructuralAtlas/'  # Hardcoded

    def __init__(self, path):
        """
        Class that contains many atlas-related things like brain region masks, names, functions to map centroids into
        binary masks, etc.

        Args:
            path: Absolute path to the folder containing all the atlas-related data which gets loaded automatically
            when an object is created.
        """
        self.path = path
        self.dimensions = np.array([974, 597, 359])
        self.shape = np.array([974, 597, 359])
        self.volume = np.product(self.dimensions)
        self.masks = load_npz(path + 'regions.npz')
        self.relativeVolumes = np.load(path + 'volumes.npy')
        self.regionNames = self.loadRegionNames(path)
        self.region_acronyms = self.load_acronyms(path)
        self.regionCentroids = self.loadRegionCentroids(path)
        self.regionCentroids['right'][57, 0] = self.dimensions[1] - self.regionCentroids['left'][57, 0]
        self.excludedRegions = []
        self.rostroCaudalOrder = self.establishRostroCaudalOrder()
        self.connectome = np.load(path + 'connectome.npy').astype('float').T
        self.directedMatrix = self.computeDirectedNetwork()
        self.undirectedMatrix = self.computeUndirectedNetwork()
        self.XYprojection = io.imread(path + 'XY_H2BGCaMP6s.tif')
        self.XZprojection = io.imread(path + 'XZ_H2BGCaMP6s.tif')

    @property
    def order(self):
        if any(self.excludedRegions):
            order = np.delete(self.rostroCaudalOrder, self.excludedRegions)
            newOrder = np.array([0] * len(order))
            for i, element in enumerate(np.unique(order)):
                newOrder[order == element] = i
            return newOrder
        else:
            return self.rostroCaudalOrder

    @property
    def names(self):
        return list(np.delete(self.regionNames, self.excludedRegions))

    @property
    def acronyms(self):
        return list(np.delete(self.region_acronyms, self.excludedRegions))

    @property
    def directed(self):
        if any(self.excludedRegions):
            return delete_rows_and_columns(self.directedMatrix, self.excludedRegions)
        else:
            return self.directedMatrix

    @property
    def undirected(self):
        if any(self.excludedRegions):
            return delete_rows_and_columns(self.undirectedMatrix, self.excludedRegions)
        else:
            return self.undirectedMatrix

    @property
    def centroids(self):
        centroids = {}
        centroids['left'] = np.delete(np.copy(self.regionCentroids['left']), self.excludedRegions, axis=0)
        centroids['right'] = np.delete(np.copy(self.regionCentroids['right']), self.excludedRegions, axis=0)
        return centroids

    def excludeRegions(self, excludedRegions):
        self.excludedRegions = excludedRegions

    def getRegionMask(self, ID, orientation='vertical'):
        if orientation == 'vertical':
            mask = np.flip(np.flip(np.flip(np.swapaxes(np.swapaxes(np.rot90(np.reshape(self.masks[:, ID].toarray(),
                                                                                       (597, 974, 359), order='F'),
                                                                            k=3), 0, 2), 1, 2), axis=1), axis=0),
                           axis=2)
        elif orientation == 'horizontal':
            mask = np.flip(np.swapaxes(np.transpose(np.reshape(self.masks[:, ID].toarray(), (597, 974, 359), order='F'),
                                                    (1, 0, 2)), 0, 2), axis=0)
        return mask

    def mapCentroids(self, centroids, orientation='vertical'):
        centroids = self.trim_centroids(centroids, orientation=orientation)
        regionLabels = np.zeros((centroids.shape[0], len(self.regionNames)))
        for i in tqdm(range(len(self.regionNames)), file=sys.stdout):
            mask = self.getRegionMask(i, orientation=orientation)
            # regionLabels[:, i] = mask[centroids[:, 1], centroids[:, 0], centroids[:, 2]]
            regionLabels[:, i] = mask[centroids[:, 2], centroids[:, 1], centroids[:, 0]]
        regionLabels = regionLabels.astype('bool')
        return regionLabels

    def mapCentroidsLeftRight(self, centroids, orientation='vertical'):
        centroids = self.trim_centroids(centroids, orientation=orientation)
        regionLabels = {}
        regionLabels['left'] = np.zeros((centroids.shape[0], len(self.regionNames)))
        regionLabels['right'] = np.zeros((centroids.shape[0], len(self.regionNames)))
        for i in tqdm(range(len(self.regionNames)), file=sys.stdout):
            mask = self.getRegionMask(i, orientation=orientation)
            for hemisphere in ['left', 'right']:
                halfMask = np.copy(mask)
                if hemisphere == 'left':
                    if orientation == 'vertical':
                        halfMask[:, :, 282:] = 0
                    elif orientation == 'horizontal':
                        halfMask[:, 282:, :] = 0
                elif hemisphere == 'right':
                    if orientation == 'vertical':
                        halfMask[:, :, :282] = 0
                    elif orientation == 'horizontal':
                        halfMask[:, :282, :] = 0
                regionLabels[hemisphere][:, i] = halfMask[centroids[:, 2], centroids[:, 1], centroids[:, 0]]
        regionLabels['left'] = regionLabels['left'].astype('bool')
        regionLabels['right'] = regionLabels['right'].astype('bool')

        return np.concatenate([regionLabels['left'], regionLabels['right']], axis=1)

    def computeDirectedNetwork(self):
        adjacency = np.copy(self.connectome)
        for i in range(adjacency.shape[0]):
            for j in range(i + 1, adjacency.shape[0]):
                adjacency[i, j] = adjacency[i, j] / (self.relativeVolumes[i] + self.relativeVolumes[j])
                adjacency[j, i] = adjacency[j, i] / (self.relativeVolumes[i] + self.relativeVolumes[j])
        adjacency[adjacency > 0] = np.log10(adjacency[adjacency > 0])
        return normalize(adjacency)

    def computeInOutDegrees(self):
        outDegrees = np.sum(self.directedMatrix, axis=0)
        inDegrees = np.sum(self.directedMatrix, axis=1)
        return inDegrees, outDegrees

    def computeUndirectedNetwork(self):
        adjacency = np.copy(self.connectome)
        for i in range(adjacency.shape[0]):
            for j in range(i + 1, adjacency.shape[0]):
                adjacency[i, j] = (adjacency[i, j] + adjacency[j, i]) / (
                        self.relativeVolumes[i] + self.relativeVolumes[j])
                adjacency[j, i] = adjacency[i, j]
        adjacency[adjacency > 0] = np.log10(adjacency[adjacency > 0])
        return normalize(adjacency)

    def computeDistanceBetweenRegions(self):
        N = self.undirectedMatrix.shape[0]
        distance = np.zeros((N, N))
        left = self.regionCentroids['left']
        right = self.regionCentroids['right']
        for i in range(N):
            for j in range(i + 1, N):
                distanceMax = np.sqrt(
                    (left[i, 0] - right[j, 0]) ** 2 + (left[i, 1] - right[j, 1]) ** 2 + (left[i, 2] - right[j, 2]) ** 2)
                distanceMin = np.sqrt(
                    (left[i, 0] - left[j, 0]) ** 2 + (left[i, 1] - left[j, 1]) ** 2 + (left[i, 2] - left[j, 2]) ** 2)
                distance[i, j] = (distanceMax + distanceMin) / 2
                distance[j, i] = distance[i, j]
        return distance

    def computeRegionCentroids(self):
        COMs = {}
        COMs['left'] = np.zeros((len(self.regionNames), 3))
        COMs['right'] = np.zeros((len(self.regionNames), 3))
        for i in range(len(self.regionNames)):
            mask = self.getRegionMask(i)
            for hemisphere in ['left', 'right']:
                halfMask = np.copy(mask)
                if hemisphere == 'left':
                    halfMask[310:, :, :] = 0
                elif hemisphere == 'right':
                    halfMask[0:311, :, :] = 0
                COMs[hemisphere][i, :] = center_of_mass(halfMask)
        COMs['right'][:, 1] = COMs['left'][:, 1]
        COMs['right'][:, 2] = COMs['left'][:, 2]
        self.regionCentroids = COMs

    def saveRegionCentroids(self):
        file = open(self.path + 'regionCentroids_structure.pkl', 'wb')
        pickle.dump(self.regionCentroids, file)
        file.close()

    def loadRegionNames(self, path):
        file = open(path + 'names.txt', 'r')
        lines = file.readlines()
        file.close()
        regions = []
        for line in lines:
            regions.append(line.split('\n')[0])
        return regions

    def load_acronyms(self, path):
        file = open(path + 'acronyms.txt', 'r')
        lines = file.readlines()
        file.close()
        regions = []
        for line in lines:
            regions.append(line.split('\n')[0])
        return regions

    def loadRegionCentroids(self, path):
        file = open(path + 'regionCentroids.pkl', 'rb')
        regionCentroids = pickle.load(file)
        file.close()
        return regionCentroids

    def establishRostroCaudalOrder(self):
        positions = []
        for region in range(len(self.regionNames)):
            # positions.append((self.regionCentroids['left'][region, 1] + self.regionCentroids['right'][region, 1]) / 2)
            positions.append(self.regionCentroids['left'][region, 1])
        order = []
        for i in range(len(self.regionNames)):
            order.append(np.where(np.flip(np.sort(positions)) == positions[i])[0][0])
        return np.array(order)

    def trim_centroids(self, centroids, orientation='vertical'):
        if orientation == 'vertical':
            centroids[:, 0] = np.clip(centroids[:, 0], 0, 596)
            centroids[:, 1] = np.clip(centroids[:, 1], 0, 973)
        elif orientation == 'horizontal':
            centroids[:, 0] = np.clip(centroids[:, 0], 0, 973)
            centroids[:, 1] = np.clip(centroids[:, 1], 0, 596)
        centroids[:, 2] = np.clip(centroids[:, 2], 0, 358)
        return centroids

    def generate_video_projection(self, timeseries, centroids, view='top', filter_noise=False, sigma=1):
        if filter:
            for i in range(timeseries.shape[0]):
                timeseries[i] = medfilt(timeseries[i], 3)
        if view == 'top':
            divider = np.zeros((self.dimensions[0], self.dimensions[1]))
            for c in centroids:
                divider[c[1], c[0]] += 1
            divider[divider == 0] = 1
            frames = []
            for i in tqdm(range(timeseries.shape[1]), file=sys.stdout):
                frame = np.zeros((self.dimensions[0], self.dimensions[1]))
                for j in range(timeseries.shape[0]):
                    frame[centroids[j][1], centroids[j][0]] += timeseries[j, i]
                frame /= divider
                frame = gaussian_filter(frame, sigma)
                frames.append(frame)
        if view == 'side':
            divider = np.zeros((self.dimensions[2], self.dimensions[1]))
            for c in centroids:
                divider[c[2], c[0]] += 1
            divider[divider == 0] = 1
            frames = []
            for i in tqdm(range(timeseries.shape[1]), file=sys.stdout):
                frame = np.zeros((self.dimensions[2], self.dimensions[1]))
                for j in range(timeseries.shape[0]):
                    frame[centroids[j][2], centroids[j][0]] += timeseries[j, i]
                frame /= divider
                frame = gaussian_filter(frame, sigma)
                frames.append(frame)
        video = np.stack(frames, axis=0)
        video -= np.amin(video)
        video /= np.amax(video)
        video *= 255
        return video

    def plot_centroids(self, ax, centroids, view='top', cmap='hot', sigma=2, vmin=0, vmax=1, s=30, alpha=0.1):
        # Loading outline
        density = np.zeros(self.dimensions)
        for c in centroids:
            density[c[1], c[0], c[2]] += 1
        # Computing density
        if view == 'top':
            density = np.sum(density, axis=2)
            density = gaussian_filter(density, sigma)
            density[density > vmax] = vmax
            density[density < vmin] = vmin
            intensity_values = density[centroids[:, 1], centroids[:, 0]]
        elif view == 'side':
            density = np.swapaxes(np.sum(density, axis=0), 0, 1)
            density = gaussian_filter(density, sigma)
            density[density > vmax] = vmax
            density[density < vmin] = vmin

            intensity_values = density[centroids[:, 2], centroids[:, 0]]
        if view == 'top':
            ax.scatter(centroids[:, 0], centroids[:, 1], s=s, alpha=alpha, c=intensity_values, cmap=cmap,
                       edgecolor='none')
        else:
            ax.scatter(centroids[:, 0], centroids[:, 2], s=s, alpha=alpha, c=intensity_values, cmap=cmap,
                       edgecolor='none')
        ax.axis('off')

    def plot_centroid_density(self, ax, centroids, view='top', cmap='hot',
                              threshold=0.02, sigma=1.5, vmin=0, vmax=0.6):
        # Loading outline
        # Computing density
        if view == 'top':
            density = np.zeros((self.dimensions[0], self.dimensions[1]))
            for c in centroids:
                density[c[1], c[0]] += 1
        elif view == 'side':
            density = np.zeros((self.dimensions[2], self.dimensions[1]))
            for c in centroids:
                density[c[2], c[0]] += 1
        density = gaussian_filter(density, sigma)
        density[density < threshold] = np.nan
        ax.imshow(density, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')

    def generate_centroid_density(self, centroids, sigma=0, crop=False):
        density = np.zeros(self.dimensions)
        density[centroids[:, 1], centroids[:, 0], centroids[:, 2]] = 1
        if sigma > 0:
            density = gaussian_filter(density, sigma)
        density = np.swapaxes(np.swapaxes(density, 0, 2), 1, 2)
        density = density.astype('double')
        density -= np.amin(density)
        density /= np.amax(density)
        density *= 65535
        # To fit with current cropped version of the atlas
        if crop:
            density = rescale_image(density, (180, 597, 974))
            density = density[:, :, 200:-57]
        return density


