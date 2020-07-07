from itertools import count, product
from functools import reduce
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation
from transformations import rotation_matrix


class Shape:
    def __init__(self, vertices, edges, dim):
        # (idx, (x, y, z, w))
        self._vertices = [(i, tuple(map(float, coords))) for i, coords in vertices]
        # Edges are tuples of indices to the vertices they connect
        self._edges = edges
        self.dim = dim
        self.transform = np.identity(dim, dtype=float)

    def set_transform(self, transform: np.ndarray):
        assert transform.shape == (self.dim, self.dim)
        self.transform = transform

    def append_transform(self, new_transform: np.ndarray):
        assert new_transform.shape == (self.dim, self.dim)
        self.transform = new_transform @ self.transform

    @property
    def vertices(self):
        return [(i, self.apply_transform(coords)) for i, coords in self._vertices]

    def apply_transform(self, coords: tuple):
        return tuple(self.transform @ np.array(coords))

    @property
    def edges(self):
        return self._edges

    def vert_coord(self, coord: int):
        return list(map(lambda x: x[1][coord], self.vertices))

    def edge_coord(self, coord: int):
        coords = []
        for edge in self.edges:
            c1, c2 = (self.vertices[edge[i]][1][coord] for i in range(2))
            coords += [c1, c2, None]
        return coords


def get_shape(shape_name='hypercube', dim=3):
    if shape_name == 'hypercube':
        vertices = list(zip(count(0), map(tuple, product(*(range(2) for _ in range(dim))))))
        edges = []
        for v1 in vertices:
            for v2 in vertices:
                if v1[0] < v2[0] and reduce(lambda a, b: a + b, [bool(v1[1][i] - v2[1][i]) for i in range(dim)], 0) == 1:
                    edges.append((v1[0], v2[0]))
        return Shape(vertices, edges, dim)


def render_shape(shape: Shape):
    fig = go.Figure(data=[go.Scatter3d(
        x=shape.vert_coord(1),
        y=shape.vert_coord(2),
        z=shape.vert_coord(3),
        mode='markers'
    ), go.Scatter3d(
        x=shape.edge_coord(1),
        y=shape.edge_coord(2),
        z=shape.edge_coord(3),
        mode='lines'
    )])
    fig.show()


if __name__ == '__main__':
    shape = get_shape(dim=4)
    shape.set_transform(rotation_matrix(70, np.array([1., 2., 3., 4.])))
    print(shape.transform)
    render_shape(shape)
