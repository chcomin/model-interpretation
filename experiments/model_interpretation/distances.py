# For distances evaluation
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_samples
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.utils.graph import _fix_connected_components
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph


def kdistance(data, n_neighbors=5, radius=None, connected=False):
    '''Calculates the geodesic distances between samples. Taken from the
    fit() method of class Isomap from scikit-learn. If `connected` is False,
    distances can be infinite.
    '''

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=radius)
    nbrs.fit(data)
    if n_neighbors is not None:
        nbg = kneighbors_graph(nbrs, n_neighbors, mode='distance')
    else:
        nbg = radius_neighbors_graph(nbrs, radius, mode='distance')

    n_connected_components, labels = connected_components(nbg)
    if connected and n_connected_components > 1:
        nbg = _fix_connected_components(
            X=nbrs._fit_X,
            graph=nbg,
            n_connected_components=n_connected_components,
            component_labels=labels
        )

    dist_matrix = shortest_path(nbg, directed=False)

    return dist_matrix

def silhouette(data, labels, use_neighbors=False, n_neighbors=5, radius=None):
    '''Calculates the silhouette coefficient for each sample.'''

    if use_neighbors:
        # Create geodesic distance matrix using isomap
        isomap = Isomap(n_neighbors=n_neighbors, radius=radius)
        isomap.fit(data)
        dists = isomap.dist_matrix_
        silhouettes = silhouette_samples(dists, labels, metric="precomputed")
    else:
        silhouettes = silhouette_samples(data, labels)

    return silhouettes