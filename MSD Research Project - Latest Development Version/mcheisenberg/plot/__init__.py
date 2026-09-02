from matplotlib.cm import ColormapRegistry
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.pyplot import Figure
from mcheisenberg.io.util import unique_path
from numpy.typing import NDArray

DEFAULT_CMAP = "viridis"

def plot3D(data: NDArray, ind: int=3, dep: int=1, cmap=DEFAULT_CMAP, size: float=0.85, show: bool=True, save: bool|str|Path=False, dir: str="out", format: str=".tiff", dpi: float=1200) -> tuple[Figure, Path|None]:
	"""
	Precondition: 1 <= ind, dep <= 3
	Precondition: data.shape[1] >= ind + dep.
	"""
	fig = plt.figure()
	plot = fig.add_subplot(projection="3d")

	# columns for independant variable(s)
	xyz = [data[:, i] for i in range(0, ind)]
	while len(xyz) < 3:
		xyz.append(np.zeros(shape=(data.shape[0],), dtype=data.dtype))

	# columns for dependant variable(s)
	uvw = [data[:, i] for i in range(ind, ind + dep)]
	if len(uvw) > 1:
		while len(uvw) < 3:
			uvw.append(np.zeros(shape=(data.shape[0],), dtype=data.dtype))

	# plot
	if len(uvw) == 1:
		plot.scatter3D(*xyz, c=data[:, -1], cmap=cmap)  # ignore size parameter
	
	else:  # len(uvw) == 3
		magnitudes = np.sqrt(sum(vec**2 for vec in uvw))
		vmin, vmax = np.min(magnitudes), np.max(magnitudes)  # shortest and longest vectors
		lerp = plt.Normalize(vmin, vmax)  # lineral interpolation
		colors = ColormapRegistry.get_cmap(cmap)(lerp(magnitudes))   # TODO: is this right?
		plot.quiver(*xyz, *uvw, colors=colors, length=size / vmax, normalize=False)

	# save as file
	path = None
	if save is not False:
		if save is True:
			path = unique_path(dir, prefix=plot3D.__name__, suffix=format)
		else:
			path = Path(path)
		
		fig.savefig(path, dpi=dpi, bbox_inches='tight')

	# display in GUI window
	if show:
		fig.show()

	return fig, path
