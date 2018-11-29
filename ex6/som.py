
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mptchs
import pickle
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from multiprocessing import cpu_count, Process, Queue


def manhatan(m, vector, shape=(10, 10)):
    dims = np.array(shape)
    delta = np.abs(m - vector)
    delta = np.where(delta > 0.5 * dims, np.abs(delta - dims), delta)
    return np.sum(delta, axis=len(m.shape) - 1)


class SOM(object):
    def __init__(self, x, y, alpha_start=0.3, seed=50):
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.sigma = x / 2.
        self.alpha_start = alpha_start
        self.alphas = None
        self.sigmas = None
        self.epoch = 0
        self.interval = int()
        self.map = np.array([])
        self.indxmap = np.stack(np.unravel_index(np.arange(x * y, dtype=int).reshape(x, y), (x, y)), 2)
        self.distmap = np.zeros((self.x, self.y))
        self.get_win_indices = np.array([])
        self.pca = None  # attribute to save potential PCA to for saving and later reloading
        self.inizialized = False
        self.error = 0.  # reconstruction error
        self.history = list()  # reconstruction error training history

    def initialize(self, inputs, how='pca'):
        self.map = np.random.normal(np.mean(inputs), np.std(inputs), size=(self.x, self.y, len(inputs[0])))
        if how == 'pca':
            eivalues = PCA(4).fit_transform(inputs.T).T
            for i in range(4):
                self.map[np.random.randint(0, self.x), np.random.randint(0, self.y)] = eivalues[i]

        self.inizialized = True

    def get_win(self, vector):
        indx = np.argmin(np.sum((self.map - vector) ** 2, axis=2))
        return np.array([indx / self.x, indx % self.y])

    def cycle(self, vector):
        w = self.get_win(vector)
        # get Manhattan distance (with PBC) of every neuron in the map to the get_win
        dists = manhatan(self.indxmap, w, self.shape)
        # smooth the distances with the current sigma
        h = np.exp(-(dists / self.sigmas[self.epoch]) ** 2).reshape(self.x, self.y, 1)
        # update neuron weights
        self.map -= h * self.alphas[self.epoch] * (self.map - vector)

        print("Epoch %i;    Neuron [%i, %i];    \tSigma: %.4f;    alpha: %.4f" %
              (self.epoch, w[0], w[1], self.sigmas[self.epoch], self.alphas[self.epoch]))
        self.epoch = self.epoch + 1

    def train(self, inputs, epochs=0, save_e=False, interval=1000, decay='hill'):
        self.interval = interval
        if not self.inizialized:
            self.initialize(inputs)
        if not epochs:
            epochs = len(inputs)
            indx = np.random.choice(np.arange(len(inputs)), epochs, replace=False)
        else:
            indx = np.random.choice(np.arange(len(inputs)), epochs)

        # get alpha and sigma decays for given number of epochs or for hill decay
        if decay == 'hill':
            epoch_list = np.linspace(0, 1, epochs)
            self.alphas = self.alpha_start / (1 + (epoch_list / 0.5) ** 4)
            self.sigmas = self.sigma / (1 + (epoch_list / 0.5) ** 4)
        else:
            self.alphas = np.linspace(self.alpha_start, 0.05, epochs)
            self.sigmas = np.linspace(self.sigma, 1, epochs)

        if save_e:  # save the error to history every "interval" epochs
            for i in range(epochs):
                self.cycle(inputs[indx[i]])
                if i % interval == 0:
                    self.history.append(self.som_error(inputs))
        else:
            for i in range(epochs):
                self.cycle(inputs[indx[i]])
        self.error = self.som_error(inputs)

    def transform(self, inputs):
        m = self.map.reshape((self.x * self.y, self.map.shape[-1]))
        dotprod = np.dot(np.exp(inputs), np.exp(m.T)) / np.sum(np.exp(m), axis=1)
        return (dotprod / (np.exp(np.max(dotprod)) + 1e-8)).reshape(inputs.shape[0], self.x, self.y)

    def compute_distance_map(self, metric='euclidean'):
        dists = np.zeros((self.x, self.y))
        for x in range(self.x):
            for y in range(self.y):
                d = cdist(self.map[x, y].reshape((1, -1)), self.map.reshape((-1, self.map.shape[-1])), metric=metric)
                dists[x, y] = np.mean(d)
        self.distmap = dists / float(np.max(dists))

    def get_win_map(self, inputs):
        wm = np.zeros(self.shape, dtype=int)
        for d in inputs:
            [(x), (y)] = self.get_win(d)
            wm[int(x), int(y)] += 1
        return wm

    def _one_get_win_neuron(self, inputs, q):
        q.put(np.array([self.get_win(d) for d in inputs], dtype='int'))

    def get_win_neurons(self, inputs):
        queue = Queue()
        n = cpu_count() - 1
        for d in np.array_split(np.array(inputs), n):
            p = Process(target=self._one_get_win_neuron, args=(d, queue,))
            p.start()
        rslt = []
        for _ in range(n):
            rslt.extend(queue.get(10))
        self.get_win_indices = np.array(rslt, dtype='int').reshape((len(inputs), 2))

    def _one_error(self, inputs, q):
        errs = list()
        for d in inputs:
            [x, y] = self.get_win(d)
            dist = self.map[int(x), int(y)] - d
            errs.append(np.sqrt(np.dot(dist, dist.T)))
        q.put(errs)

    def som_error(self, inputs):
        queue = Queue()
        for d in np.array_split(np.array(inputs), cpu_count()):
            p = Process(target=self._one_error, args=(d, queue,))
            p.start()
        rslt = []
        for _ in range(cpu_count()):
            rslt.extend(queue.get(10))
        return sum(rslt) / float(len(inputs))

    def plot_point_map(self, inputs, targets, targetnames, filename=None, colors=None, markers=None, mol_dict=None,
                       density=True, activities=None):
        if not markers:
            markers = ['o'] * len(targetnames)
        if not colors:
            colors = ['#EDB233', '#90C3EC', '#C02942', '#79BD9A', '#774F38', 'gray', 'black']
        if activities:
            heatmap = plt.get_cmap('coolwarm').reversed()
            colors = [heatmap(a / max(activities)) for a in activities]
        if density:
            fig, ax = self.plot_density_map(inputs, internal=True)
        else:
            fig, ax = plt.subplots(figsize=self.shape)

        for cnt, xx in enumerate(inputs):
            if activities:
                c = colors[cnt]
            else:
                c = colors[targets[cnt]]
            w = self.get_win(xx)
            ax.plot(w[1] + .5 + 0.1 * np.random.randn(1), w[0] + .5 + 0.1 * np.random.randn(1),
                    markers[targets[cnt]], color=c, markersize=12)

        ax.set_aspect('equal')
        ax.set_xlim([0, self.x])
        ax.set_ylim([0, self.y])
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        ax.grid(which='both')

        if not activities:
            patches = [mptchs.Patch(color=colors[i], label=targetnames[i]) for i in range(len(targetnames))]
            legend = plt.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(targetnames),
                                mode="expand", borderaxespad=0.1)
            legend.get_frame().set_facecolor('#e5e5e5')

        if mol_dict:
            for k, v in mol_dict.items():
                w = self.get_win(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='inputs', fontsize=18, fontweight='bold')

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Point map plot done!")
        else:
            plt.show()

    def plot_density_map(self, inputs, colormap='Oranges', filename=None, mol_dict=None, internal=False):
        wm = self.get_win_map(inputs)
        fig, ax = plt.subplots(figsize=self.shape)
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        ax.set_aspect('equal')

        if mol_dict:
            for k, v in mol_dict.items():
                w = self.get_win(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='inputs', fontsize=18, fontweight='bold')

        if not internal:
            if filename:
                plt.savefig(filename)
                plt.close()
                print("Density map plot done!")
            else:
                plt.show()
        else:
            return fig, ax
