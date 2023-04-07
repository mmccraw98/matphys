import numpy as np
import pandas as pd
import os
from data_manipulation import get_files, ibw2dict, ibw2df
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from mlinsights.mlmodel import KMeansL1L2
from scipy.optimize import minimize, curve_fit
from utils.generic_funcs import progress_bar

def format_fd(df, k):
    '''
    Format force-distance data from an AFM experiment into more useful units.
    '''
    f = df.Defl.values * k
    h = (df.ZSnsr - df.Defl).values
    j0 = np.argmax(df.ZSnsr.values)
    j = np.argmax(f[: j0])
    i = np.argmin(f[:j])
    f = f[i:j] - f[i]
    h = h[i:j] - h[i]
    return pd.DataFrame({'f': abs(f), 'h': abs(h)})


def get_line_point_coords(filename):
    """get the line and point coordinates from the filename

    Args:
        filename (str): name of the file

    Returns:
        tuple: line, point coordinates
    """
    just_the_name = os.path.split(filename)[-1].split('.')[0]
    line, point = just_the_name.strip('Line').split('Point')
    return int(line), int(point)


def merge_fmap_masks(maps):
    return np.prod([m.feature_mask for m in maps], axis=0)


class ForceMap:

    def __init__(self, root_directory, spring_const=None, sampling_frequency=None, probe_radius=1, contact_beta=3 / 2,
                 pct_smooth=0.01):
        self.root_directory = root_directory
        self.map_directory = None
        self.shape = None
        self.dimensions = None
        self.spring_const = spring_const
        self.sampling_frequency = sampling_frequency
        self.contact_alpha = 16 * np.sqrt(probe_radius) / 3
        self.contact_beta = contact_beta
        self.map_scalars = {}
        self.map_vectors = None
        self.fd_curves = None
        self.x = None
        self.y = None
        self.pct_smooth = pct_smooth
        self.feature_mask = None

        self.load_map()

    def load_map(self):
        # find files
        files = get_files(self.root_directory)
        # find the map file
        possible_map = [file for file in files if not all(kw in file for kw in ['Line', 'Point'])]
        if len(possible_map) != 1:
            exit('the .ibw file for the map height data is missing or duplicated')
        self.map_directory = possible_map[0]

        print('loading map data...', end='\r')
        map_dict = ibw2dict(self.map_directory)
        if self.spring_const is None:
            self.spring_const = map_dict['notes']['SpringConstant']
        self.dimensions = np.array([map_dict['notes']['ScanSize'], map_dict['notes']['ScanSize']])
        for i, label in enumerate(map_dict['labels']):
            data = np.array(map_dict['data'])[:, :, i]
            if i == 0:
                self.shape = data.shape
                x, y = np.linspace(0, self.dimensions[0], self.shape[0]), np.linspace(0, self.dimensions[1],
                                                                                      self.shape[1])
                self.x, self.y = np.meshgrid(x, y)
            self.map_scalars.update({label: data.T})
        print('done', end='\r')

        self.map_vectors = np.zeros(self.shape, dtype='object')
        self.fd_curves = np.zeros(self.shape, dtype='object')
        self.feature_mask = np.ones(self.shape) == 1
        for i, file in enumerate(files):
            progress_bar(i, len(files) - 1, message='loading force curves')
            if file == self.map_directory:
                continue
            coords = get_line_point_coords(file)
            self.map_vectors[coords] = ibw2df(file)
        print('done', end='\r')

    def transpose(self):
        for key, value in self.map_scalars.items():
            self.map_scalars[key] = value.T[::-1]
        self.feature_mask = self.feature_mask.T[::-1]
        self.map_vectors = self.map_vectors.T[::-1]
        self.fd_curves = self.fd_curves.T[::-1]

    def plot_map(self):
        figs, axs = plt.subplots(1, len(self.map_scalars.keys()))
        for i, (ax, key) in enumerate(zip(axs, self.map_scalars.keys())):
            a_i = ax.contourf(self.x, self.y, self.map_scalars[key])
            ax.set_title(key)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            figs.colorbar(a_i, cax=cax, orientation='vertical')
        plt.tight_layout()
        plt.show()

    # TODO make 3d surface plot

    def flatten_and_shift(self, order=1, left_right_mask=[0, 1], up_down_mask=[0, 1], show_plots=True):
        # l1 optimization of background shift to minimize outlier error
        def obj(X, func, real, mask):
            return np.sum(abs(func(X) - real) * mask ** 2)

        if order not in [0, 1, 2]:
            exit('flattening {} doesnt make sense to me so i will crash now :)'.format(order))
        if 'MapHeight' not in self.map_scalars.keys():
            exit('there is no height data')
        height = self.map_scalars['MapHeight'].copy()

        mask = np.ones(height.shape)
        mask[int(up_down_mask[0] * mask.shape[0]): int(up_down_mask[1] * mask.shape[0]),
        int(left_right_mask[0] * mask.shape[1]): int(left_right_mask[1] * mask.shape[1])] = 0

        if show_plots:
            plt.imshow(mask)
            plt.title('Mask')
            plt.show()

        if order == 0:
            height -= np.min(height[mask == 1])

        elif order == 1:
            def lin(X):
                A, B, C = X
                return A * self.x + B * self.y + C

            x_opt = minimize(obj, x0=[0, 0, 0], args=(lin, height, mask), method='Nelder-Mead').x
            height -= lin(x_opt)

        elif order == 2:
            def quad(X):
                A, B, C, D, E = X
                return A * self.x ** 2 + B * self.x + C * self.y ** 2 + D * self.y + E

            x_opt = minimize(obj, x0=[0, 0, 0, 0, 0], args=(quad, height, mask), method='Nelder-Mead').x
            height -= quad(x_opt)

        self.map_scalars.update({'MapFlattenHeight': height - np.min(height)})

    def format_fds(self):
        tot = 0
        for i, row in enumerate(self.map_vectors):
            for j, df in enumerate(row):
                tot += 1
                progress_bar(tot, self.shape[0] * self.shape[1], message='formatting force curves')
                self.fd_curves[i, j] = format_fd(df, self.spring_const, self.pct_smooth)
        print('done', end='\r')

    def cut_background(self, mult, show_plots=True):
        height_map = self.map_scalars['MapFlattenHeight'].copy()
        heights = height_map.ravel()
        cut = np.mean(heights) * mult
        self.feature_mask = height_map > cut
        if show_plots:
            figs, axs = plt.subplots(1, 2)
            axs[0].hist(heights, bins=100, label='Full')
            axs[0].hist(heights[heights > cut], bins=100, label='Cut')
            axs[0].legend()
            axs[0].set_title('Height Histogram')
            height_map[np.invert(self.feature_mask)] = -1
            masked_array = np.ma.masked_where(height_map == -1, height_map)
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='white')
            axs[1].contourf(self.x, self.y, masked_array, cmap=cmap)
            axs[1].set_xlabel('x (m)')
            axs[1].set_ylabel('y (m)')
            axs[1].set_title('Masked')
            plt.tight_layout()
            plt.show()

    def ml_flatten_and_shift(self, num_features=2, order=1, show_plots=False):
        '''
        kind of experimental function for flattening the height of force maps and shifting them to start at 0 height
        there are assumed to be a certain number features in an image for instance, a cell sitting atop a culture dish
        gives two features: the cell and the dish.  we can then identify these features by their distinct heights
        (we use 2 by default) and then we take the lowest height out of the group and make a mask
        using the mask, we fit a surface of a given order to the mask and then subtract the fitted surface from the
        height data
        :param num_features: number of distinct topographical features in the force map
        :param order: order of surface plane fit for background subtraction
        :param show_plots: whether or not to show the mask image
        :return: adds a mapflattenheight element to self.map_scalars associated with the corrected height map
        and adds a feature_map to self corresponding to the non-cut portion of the height map
        '''

        # l1 optimization of background shift to minimize outlier error
        def obj(X, func, real, mask):
            return np.sum(abs(func(X) - real) * mask ** 2)

        if order not in [0, 1, 2]:
            exit('flattening {} doesnt make sense to me so i will crash now :)'.format(order))
        if 'MapHeight' not in self.map_scalars.keys():
            exit('there is no height data')

        self.format_fds()
        sizes = np.array([df.f.size for df in self.fd_curves.ravel()])
        height = self.map_scalars['MapHeight'].copy().ravel()
        model = KMeansL1L2(n_clusters=num_features, norm='L1', init='k-means++', random_state=42)
        data = np.concatenate([feature.reshape(-1, 1) for feature in [height, sizes]], axis=1)
        model.fit(data)
        background_label = np.argmin([np.mean(height[model.labels_ == label]) for label in np.unique(model.labels_)])
        mask = np.invert(model.labels_ == background_label).reshape(self.shape)
        height = height.reshape(self.shape)
        if show_plots:
            plt.imshow(mask)
            plt.title('Mask')
            plt.show()
        if order == 0:
            height -= np.min(height[np.invert(mask)])

        elif order == 1:
            def lin(X):
                A, B, C = X
                return A * self.x + B * self.y + C

            x_opt = minimize(obj, x0=[0, 0, 0], args=(lin, height, np.invert(mask)), method='Nelder-Mead').x
            height -= lin(x_opt)

        elif order == 2:
            def quad(X):
                A, B, C, D, E = X
                return A * self.x ** 2 + B * self.x + C * self.y ** 2 + D * self.y + E

            x_opt = minimize(obj, x0=[0, 0, 0, 0, 0], args=(quad, height, np.invert(mask)), method='Nelder-Mead').x
            height -= quad(x_opt)

        self.map_scalars.update({'MapFlattenHeight': height - np.min(height)})
        self.feature_mask = mask

    def copy(self):
        return deepcopy(self)

    # TODO thin sample correction

    # TODO tilted sample correction


def offset_polynom(x, y_offset, x_offset, slope):
    # model of an offset polynomial with power 2 (3/2 doesn't seem to work!)
    poly = 2
    d = x - x_offset
    return slope * (abs(d) - d) ** poly + y_offset

def fit_contact(df, k):
    # fit QUADRATIC f-z model to data and return contact location and index
    # format f-d dataframe values and cut to maximum
    f = df.Defl.values * k
    i = np.argmax(f)
    z_tip = (df.Defl - df.ZSnsr).values[:i]
    f = f[:i]

    # normalize f-d values
    x = z_tip - z_tip[0]
    x /= max(abs(x))
    y = f - f[0]
    y /= max(abs(f))

    # fit to the offset polynomial model
    X, _ = curve_fit(offset_polynom, x, y)
    y_offset, x_offset, slope = X

    # reverse the normalization and get index
    z_offset = x_offset * max(abs(z_tip - z_tip[0])) + z_tip[0]
    j = np.argmin((z_tip - z_offset) ** 2)
    return z_offset, j

