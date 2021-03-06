
import numpy as np
from sklearn.base import BaseEstimator
#import matplotlib.pyplot as plt
#import tensorflow as tf
#from .utils import InputError, is_positive_integer, ceil
#from sklearn.manifold import *
#from sklearn.decomposition import *
#from sklearn.linear_model import *
#from sklearn.model_selection import *
#from sklearn.discriminant_analysis import *
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy import interpolate
#tfd = tf.contrib.distributions
#from qml.utils import is_none
from qml.qmlearn.data import Data
from qml.qmlearn.representations import GlobalSLATM as qmlGlobalSLATM
from qml.qmlearn.representations import AtomicSLATM as qmlAtomicSLATM
import rmsd
from msmbuilder.decomposition import tICA as msmtICA

class tICA(msmtICA):

    def __init__(self, **kwargs):
        super(tICA, self).__init__(**kwargs)

    def fit(self, X, y=None):
        if isinstance(X, list):
            return msmtICA.fit(self, X, y)
        else:
            return msmtICA.fit(self, [X], y)

    def transform(self, X):
        if isinstance(X, list):
            return msmtICA.transform(self, X)
        else:
            return msmtICA.transform(self, [X])[0]

def colorplot(x, y=None, imgname = None, same_axis = True):
    """
    Create a 2D plot or 1D if y == None
    """
    if y is None:
        plt.figure(figsize=(10,5))
        # Do 1D plot
        plt.plot(range(len(x)), x)
    else:
        # Do 2D plot

        # Create figure
        fig = plt.figure(figsize=(10,15))
        gs = GridSpec(2, 1, height_ratios=[2, 1]) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # ax0
        # fit spline
        tck,u=interpolate.splprep([x, y],s=0.0)
        x_i,y_i= interpolate.splev(np.linspace(0,1,10000),tck)

        # Gradient color change magic
        z = np.linspace(0.0, 1.0, x_i.shape[0])
        points = np.array([x_i,y_i]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, array=z, cmap = 'viridis',
                norm = plt.Normalize(0.0, 1.0), alpha = 0.8)
        ax0.add_collection(lc)

        # plotting
        if same_axis:
            max_ = max(x_i.max(), y_i.max())
            min_ = min(x_i.min(), y_i.min())
            range_ = max_ - min_
            ax0.set_xlim([min_-0.1*range_, max_+0.1*range_])
            ax0.set_ylim([min_-0.1*range_, max_+0.1*range_])
        else:
            xrange_ = x_i.max() - x_i.min()
            yrange_ = y_i.max() - y_i.min()
            ax0.set_xlim([x_i.min()-0.1*xrange_, x_i.max()+0.1*xrange_])
            ax0.set_ylim([y_i.min()-0.1*yrange_, y_i.max()+0.1*yrange_])

        # time plots
        ax1.plot(range(len(x)), x)
        ax1.plot(range(len(y)), y)

    if imgname is None:
        plt.show()
    else:
        plt.savefig(imgname + ".png", dpi = 600, bbox_inches='tight')
        plt.clf()

def significance(data, indices, m):
    """
    Get the norm of the numerical gradients of the output of
    a sklearn model `m` wrt carteesian coordinates given by indices.
    """

    def make_pertubations(data, indices, deviation = 1e-4):
        """
        Create small deviations to the coordinates along
        the x, y and z coordinate
        """

        objects = []
        for i in indices:
            data_object = Data()
            nuclear_charges = data.nuclear_charges[i]
            xyz = data.coordinates[i]
            data_object.ncompounds = 4 * xyz.size
            data_object.nuclear_charges = np.asarray([nuclear_charges for _ in range(data_object.ncompounds)])
            data_object.natoms = np.asarray([nuclear_charges.size] * 4 * xyz.size)

            coords = np.empty([4*xyz.size, xyz.size], dtype = float)
            xyz_flat = xyz.ravel()
            zeros = np.zeros(xyz.size)

            c = 0
            for i in range(xyz.size):
                dev = zeros.copy()
                dev[i] += deviation
                coords[c] = xyz_flat - 2*dev
                coords[c + 1] = xyz_flat - dev
                coords[c + 2] = xyz_flat + dev
                coords[c + 3] = xyz_flat + 2*dev
                c += 4

            data_object.coordinates = coords.reshape((4*xyz.size,) + xyz.shape)
            objects.append(data_object)

        return objects

    def calculate_numerical_gradient(y, deviation = 1e-4):
        """
        Get partial gradients from function values
        """
        mol_grad = []

        x_size = y.shape[1]//4
        x_shape = (x_size // 3, 3)

        grad_shape = (y.shape[0], y.shape[-1], x_size)
        gradients = np.zeros(grad_shape)

        for c, i in enumerate(range(0, y.shape[1], 4)):
            grad = (-y[:,i+3] + 8*y[:,i+2] - 8*y[:,i+1] + y[:,i]) / (12 * deviation)
            gradients[:,:,c] = grad

        return gradients.reshape((y.shape[0], y.shape[-1]) + x_shape)


    if isinstance(indices, int):
        indices = [indices]

    perturbed_objects = make_pertubations(data, indices)
    values = np.asarray([m.transform(data_object) for data_object in perturbed_objects])
    partial_grad = calculate_numerical_gradient(values)
    # Get percentage of norm per atom
    x = np.sqrt(np.sum(partial_grad**2, axis=3))
    return x / x.sum(2)[:,:,None]

class Align(BaseEstimator):
    """
    Align molecules and get their euclidian distance to some basis molecules
    """

    def __init__(self, basis):
        """
        :param basis: indices of the molecules that will act as basis
        :type basis: list
        """

        self.basis = basis


    def fit(self, X, y=None):

        self.basis_coords = X.coordinates[self.basis]
        self.basis_coords -= np.mean(self.basis_coords, axis=1)[:,None]

        return self

    def transform(self, X):

        # Align molecules P to molecules Q
        P = X.coordinates
        P -=  np.mean(P, axis=1)[:,None]
        Q = self.basis_coords


        C = np.dot(np.transpose(P, [0,2,1]), Q.T)
        V, S, W = np.linalg.svd(np.transpose(C, [0,3,1,2]))
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
        S[d,-1] *= -1
        V[d, :, -1] *= -1

        U = np.dot(V[0,1], W[0,1])
        U = np.matmul(V, W)

        P = np.einsum('ijk,iabk->ijba', P, U)

        delta = P - np.transpose(Q, [1,2,0])

        rmsd = np.sqrt(np.sum(delta**2, axis=(1,2)))

        return rmsd

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class Coordinates(BaseEstimator):
    """
    Just return the coordinates ravelled.
    """

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        if hasattr(X, 'indices'):
            return X.coordinates[X.indices].reshape(X.indices.size, -1)
        else:
            return X.coordinates.reshape(X.coordinates.shape[0], -1)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class Distances(BaseEstimator):
    """
    Upper triangle of internal distances between atoms.
    """

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        return self._transform(X)

    def _transform(self, X):

        if hasattr(X, 'indices'):
            P = X.coordinates[X.indices]
        else:
            P = X.coordinates


        distance_matrix = np.sqrt(np.sum((P[:,:,None] - P[:,None,:]) ** 2, axis = 3))

        iut = np.triu_indices(P.shape[1], 1)
        return distance_matrix[:,iut[0], iut[1]]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class InverseDistances(Distances):
    """
    Upper triangle of internal inverse distances between atoms.
    """

    def transform(self, X):
        return 1 / self._transform(X)

class GlobalSLATM(qmlGlobalSLATM):

    def __init__(self, **params):
        super(GlobalSLATM, self).__init__(**params)

    def transform(self, X):
        return qmlGlobalSLATM.transform(self, X).representations

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class AtomicSLATM(qmlAtomicSLATM):

    def __init__(self, indices = None, **params):
        self.indices = indices
        super(AtomicSLATM, self).__init__(**params)

    def transform(self, X):
        representations = qmlAtomicSLATM.transform(self, X).representations[:,self.indices]
        # Concatenate all the atomic representations
        return representations.reshape(representations.shape[0], -1)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
# https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_tensorflow.py
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/
#class VAE(object):
#    """
#    Dimensionality reduction with variational autoencoders.
#    The code is slightly modified from 
#    https://danijar.com/building-variational-auto-encoders-in-tensorflow/
#
#    """
#
#
#    def __init__(self, dimensions = 2, layer_sizes = [200,200], learning_rate = 0.001,
#            activation_function = tf.nn.relu, n_samples = 1, n_iterations = 20, batch_size = 100,
#            filenames = []):
#        self.dimensions = dimensions
#        self.layer_sizes = layer_sizes
#        self.learning_rate = learning_rate
#        self.activation_function = activation_function
#        self.n_samples = n_samples
#        self.n_iterations = n_iterations
#        self.batch_size = batch_size
#        self._generate_compounds(filenames)
#
#    def _get_batch_size(self):
#        """
#        Determines the actual batch size.
#        If the batch size is larger than the number of samples, it is truncated and a warning
#        is printed.
#
#        Furthermore the returned batch size will be slightly modified from the user input if
#        the last batch would be tiny compared to the rest.
#
#        :return: Batch size
#        :rtype: int
#        """
#
#        if self.batch_size > self.n_samples:
#            print("Warning: batch_size larger than sample size. It is going to be clipped")
#            return min(self.n_samples, self.batch_size)
#        else:
#            batch_size = self.batch_size
#
#        # see if the batch size can be modified slightly to make sure the last batch is similar in size
#        # to the rest of the batches
#        # This is always less that the requested batch size, so no memory issues should arise
#        better_batch_size = ceil(self.n_samples, ceil(self.n_samples, batch_size))
#
#        return better_batch_size
#
#    def _make_encoder(self, data):
#        """
#        ReLU activation on hidden layers is used and softplus
#        to force scale to be positive.
#        """
#        x = tf.layers.flatten(data)
#        for layer_size in self.layer_sizes:
#            x = tf.layers.dense(x, layer_size, self.activation_function)
#
#        loc = tf.layers.dense(x, self.dimensions)
#        scale = tf.layers.dense(x, self.dimensions, tf.nn.softplus)
#        return tfd.MultivariateNormalDiag(loc, scale), loc
#
#    def _make_prior(self):
#        """
#        This could in principle be changed to a different distribution
#        if we want a more uniform distribution or something
#        distributed as a square instead of a circle (in 2D)
#        """
#        loc = tf.zeros(self.dimensions)
#        scale = tf.ones(self.dimensions)
#        return tfd.MultivariateNormalDiag(loc, scale)
#
#    def _make_decoder(self, z):
#        """
#        Decoder uses the same layer_sizes as the encoder,
#        but this is not necessarily optimal.
#        ReLU is applied on the output layer since all input
#        features are assumed positive, but exp could be used
#        as well.
#        Returns a gamma distribution since input is positive or zero.
#        A mixture might be better since the input is sparse.
#        Alternatively a beta-like distribution could be used if the max/min of the output
#        was stored or the input was normalised.
#
#        """
#
#        for layer_size in self.layer_sizes:
#            z = tf.layers.dense(z, layer_size, self.activation_function)
#
#        #x = tf.layers.dense(x, self.n_features, tf.nn.relu)
#        alpha = tf.layers.dense(z, self.n_features, tf.nn.softplus)
#        beta = tf.layers.dense(z, self.n_features, tf.nn.softplus)
#        return tfd.Independent(tfd.Gamma(alpha, beta), 1)
#
#    def _generate_compounds(self, filenames):
#        """
#        Creates QML compounds.
#
#        :param filenames: path of xyz-files
#        :type filenames: list
#        """
#
#        self.compounds = np.empty(len(filenames), dtype=object)
#        for i, filename in enumerate(filenames):
#            self.compounds[i] = Compound(filename)
#
#
#    def _get_slatm(self):
#        mbtypes = self._get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
#        x1 = np.empty(len(self.compounds), dtype=object)
#        x2 = np.empty(len(self.compounds), dtype=object)
#        coords = []
#        for i, mol in enumerate(self.compounds):
#            mol.generate_slatm(mbtypes, local = False)
#            x1[i] = mol.representation
#            mol.generate_coulomb_matrix(size = 20, sorting = "unsorted")
#            x2[i] = mol.representation
#            coords.append(mol.coordinates)
#        x1 = np.asarray(list(x1), dtype=float)
#        x2 = np.asarray(list(x2), dtype=float)
#        coords = np.asarray(coords)
#
#        return x1, x2, coords
#
#    def _get_slatm_mbtypes(self, arr):
#        from qml.representations import get_slatm_mbtypes
#        return get_slatm_mbtypes(arr)
#
#    def fit(self):
#        slatm, cm, coords = self._get_slatm()
#
#        d1 = np.sum((coords[:,0] - coords[:,3])**2, axis = 1)
#        d2 = np.sum((coords[:,4] - coords[:,3])**2, axis = 1)
#
#        n = len(self.compounds)
#        idx = np.random.randint(0, n, size = 100)
#        range_ = np.arange(n)
#
#        # Remove constant features
#        slatm = slatm[:, slatm.std(0) > 1e-6]
#        #slatm_test = slatm[idx]
#        #slatm_train = slatm[~np.isin(range_,idx)]
#
#        cm = cm[:, cm.std(0) > 1e-6]
#        #cm_test = cm[idcm]
#        #cm_train = cm[~np.isin(range_,idcm)]
#
#        cm0 = cm[:1000]
#        cm1 = cm[-1000:]
#        cmc = np.concatenate([cm0,cm1], axis=0)
#        slatm0 = slatm[:1000]
#        slatm1 = slatm[-1000:]
#        slatmc = np.concatenate([slatm0,slatm1], axis=0)
#        y = np.zeros(2000, dtype=int)
#        y[-1000:] = 1
#
#        D2 = d2[1020:1100]
#        D1 = d1[1020:1100]
#        Dcm = cm[1020:1100]
#        Dslatm = slatm[1020:1100]
#
#        """
#        Run a classifier and use probabilities.
#        Alternatively run regressor and modify hyperparams
#        to make the transition from one state as sharp as needed.
#        """
#        #mod = LogisticRegression()
#        #dist = {"C": 10**np.linspace(-1, 2, 1000),
#        #        "penalty": ["l1", "l2"]}
#        #cv_gen = RandomizedSearchCV(mod, dist, cv = 5, verbose = 1, n_iter = 20, refit=False)
#        #cv_gen.fit(cmc,y)
#        #print(cv_gen.best_params_, cv_gen.best_score_)
#        #mod.set_params(**cv_gen.best_params_)
#        ##mod = CalibratedClassifierCV(mod, cv=3, method = "sigmoid")
#        #mod.fit(cmc,y)
#        #pred = mod.predict_proba(Dcm)
#        #colorplot(range(1020,1100), pred[:,0], "class_cm")
#        #cv_gen.fit(slatmc,y)
#        #print(cv_gen.best_params_, cv_gen.best_score_)
#        #mod.set_params(**cv_gen.best_params_)
#        ##mod = CalibratedClassifierCV(mod, cv=3, method = "sigmoid")
#        #mod.fit(slatmc,y)
#        #pred = mod.predict_proba(Dslatm)
#        #colorplot(range(1020,1100), pred[:,0], "class_slatm")
#
#
#
#        # Classic
#        colorplot(D2, D1, "classic")
#
#        # Dim red
#        mod = LocallyLinearEmbedding(n_neighbors = 5, n_components = 2, method = "ltsa")
#        y = mod.fit_transform(Dcm)
#        colorplot(y[:,0], y[:,1], "ltsa_cm")
#        mod = LocallyLinearEmbedding(n_neighbors = 7, n_components = 2, method = "ltsa")
#        y = mod.fit_transform(Dslatm)
#        colorplot(y[:,0], y[:,1], "ltsa_slatm")
#        #mod = LocallyLinearEmbedding(n_neighbors = 5, n_components = 2, method = "modified")
#        #y = mod.fit_transform(Dcm)
#        #colorplot(y[:,0], y[:,1], "mod_cm")
#        #mod = LocallyLinearEmbedding(n_neighbors = 7, n_components = 2, method = "modified")
#        #y = mod.fit_transform(Dslatm)
#        #colorplot(y[:,0], y[:,1], "mod_slatm")
#        mod = MDS(n_components = 2, n_init = 10)
#        y = mod.fit_transform(Dcm)
#        colorplot(y[:,0], y[:,1], "mds_cm")
#        mod = MDS(n_components = 2, n_init = 10)
#        y = mod.fit_transform(Dslatm)
#        colorplot(y[:,0], y[:,1], "mds_slatm")
#        mod = PCA(n_components = 2)
#        y = mod.fit_transform(Dcm)
#        colorplot(y[:,0], y[:,1], "pca_cm")
#        mod = PCA(n_components = 2)
#        y = mod.fit_transform(Dslatm)
#        colorplot(y[:,0], y[:,1], "pca_slatm")
#
#        quit()
#
#        return self._fit(x_train, x_test)
#
#    def _fit(self, x, x_test = None):
#        self.n_samples = x.shape[0]
#        self.n_features = x.shape[1]
#        batch_size = self._get_batch_size()
#        n_batches = ceil(self.n_samples, batch_size)
#
#        data = tf.placeholder(tf.float32, [None, self.n_features])
#
#        # Create the dataset iterator
#        dataset = tf.data.Dataset.from_tensor_slices(data)
#        dataset = dataset.shuffle(buffer_size = self.n_samples)
#        dataset = dataset.batch(batch_size)
#        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
#        tf_data = iterator.get_next()
#
#        make_encoder = tf.make_template('encoder', self._make_encoder)
#        make_decoder = tf.make_template('decoder', self._make_decoder)
#
#        # Define the model.
#        prior = self._make_prior()
#        posterior, post_means = make_encoder(data)
#        code = posterior.sample()#self.n_samples)
#
#        # Define the loss.
#        # likelihood ~ E[log(P(X|z))]. This is approximated by a single sample
#        # of z. In principle we could draw several samples to get a better approximation.
#        likelihood = make_decoder(code).log_prob(data)
#        # divergence ~ D_KL[Q(z|X) || P(z)] 
#        divergence = tfd.kl_divergence(posterior, prior)
#        # elbo ~ lower bound of log(P(X))
#        elbo = tf.reduce_mean(likelihood / self.n_samples - divergence)
#        optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(-elbo)
#
#        #samples = make_decoder(prior.sample(10), [28, 28]).mean()
#
#        # Initialize
#        init = tf.global_variables_initializer()
#        iterator_init = iterator.make_initializer(dataset)
#        quit()
#
#        with tf.train.MonitoredSession() as sess:
#            sess.run(init)
#            for it in range(self.n_iterations):
#                test_cost = sess.run(elbo, feed_dict = {data:x_test})
#                print("Test cost at iteration %d: %6.3f" % (it+1, test_cost))
#                sess.run(iterator_init, feed_dict={data:x})
#                for _ in range(n_batches):
#                    sess.run(optimize)
#            sess.run(iterator_init, feed_dict={data:x})
#            z_means = []
#            for _ in range(n_batches):
#                z_means_batch = sess.run(post_means)
#                z_means.append(z_means_batch)
#
#        return z_means
#
