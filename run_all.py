import ddr
import numpy as np
import glob
from qml.qmlearn.data import Data
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA, TruncatedSVD, NMF, \
        LatentDirichletAllocation, IncrementalPCA, FastICA, FactorAnalysis
from sklearn.manifold import Isomap, TSNE, MDS

METHODS        = {'pca': PCA,
                  'tsvd': TruncatedSVD,
                  'nmf': NMF,
                  'lda': LatentDirichletAllocation,
                  'ipca': IncrementalPCA,
                  'fastica': FastICA,
                  'factor': FactorAnalysis,
                  'isomap': Isomap,
                  'tsne': TSNE,
                  'mds': MDS,
                  'tica': ddr.tICA, # tica is the only time series
                  }


def make_classic(data, coord1, coord2, coord3):
    """
    Use pairs of interatomic distances as reaction coordinates.
    coord1 is the index of the atom being transfered.
    coord2 and coord3 are the two atoms whose distance to coord1
    are the reaction coordinates
    """

    d1 = np.sum((data.coordinates[:,coord1] - data.coordinates[:,coord2])**2, axis=1)**0.5
    d2 = np.sum((data.coordinates[:,coord1] - data.coordinates[:,coord3])**2, axis=1)**0.5
    ddr.colorplot(d1,d2, 'classic')

def make_align_method(data, dim=2, indices=[0], method='pca'):
    """
    Calculate the aligned distance between all given molecules and the molecules
    defined by the given indices as basis functions. Then run a dimensionality
    reduction method defined in METHODS.
    """

    # Align all the molecules. Use the given indices
    # as basis functions
    align = ddr.Align(indices)
    # Dimensionality reduction. Can add more options here.
    decomp = METHODS[method](n_components = dim)
    # Put them together in a pipeline
    m = make_pipeline(align, decomp)
    # Transform some data
    reduced_space = m.fit_transform(data)

    if dim == 2:
        ddr.colorplot(reduced_space[:,0], reduced_space[:,1], 'align_%s' % method)
    elif dim == 1:
        ddr.colorplot(reduced_space, 'align_%s' % method)
    else:
        print("Warning: Plotting only supports up to 2D")

def make_distance_method(data, dim=2, method='pca'):
    """
    Calculate the aligned distance between all given molecules and the molecules
    defined by the given indices as basis functions. Then run a dimensionality
    reduction method defined in METHODS.
    """

    # Align all the molecules. Use the given indices
    # as basis functions
    align = ddr.InverseDistances()
    # Dimensionality reduction. Can add more options here.
    decomp = METHODS[method](n_components = dim)
    # Put them together in a pipeline
    m = make_pipeline(align, decomp)
    # Transform some data
    reduced_space = m.fit_transform(data)

    if dim == 2:
        ddr.colorplot(reduced_space[:,0], reduced_space[:,1], 'distance_%s' % method)
    elif dim == 1:
        ddr.colorplot(reduced_space, 'distance_%s' % method)
    else:
        print("Warning: Plotting only supports up to 2D")

    s = ddr.significance(data, [0,170,349], m)
    print((s[0,0]))
    print((s[0,1]))
    print((s[1,0]))
    print((s[1,1]))
    print((s[2,0]))
    print((s[2,1]))

if __name__ == '__main__':

    # H3 is taken off C4 by C0
    # The reaction happens around frame 7570 or so
    filenames = sorted(glob.glob('data/hexane/hex_*.xyz'))[7400:7750]
    data = Data(filenames)
    # Use interatomic distances between atoms 0-3 and 0-4.
    #make_classic(data, 3, 4, 0)
    # Align molecules to a set of basis molecules and apply dimensionality reduction
    #make_align_method(data, dim=2, indices=np.arange(160,180), method='pca')
    make_distance_method(data, dim=2, method='pca')


#    s = ddr.significance(data, [0,170,349], m)
#    print((s[0,0]))
#    print((s[0,1]))
#    print((s[1,0]))
#    print((s[1,1]))
#    print((s[2,0]))
#    print((s[2,1]))


