import __future__
import numpy as np
from scipy.signal import fftconvolve
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from numpy.typing import NDArray

def acf(signal: NDArray, mask: NDArray=None) -> NDArray:
	"""
	Computes the (normalized) autocorrelation function of the given signal.

	@param signal - A single (potential multidimensional) discrete finite signal/sequence.

	@param mask - (Optional) mask defining irregular boundaries.
		Precondition: all elements are either 1.0 (True) or 0.0 (False).
		Precondition: mask.shape == signal.shape
	
	@return All elements are well defined regaurdless of mask.
		Postcondition: acf().shape == signal.shape
	"""
	_REVERSED = (slice(None, None, -1),) * signal.ndim  # i.e. [::-1, ::-1, ...] for all axes in signal
	_HALF = tuple(slice(n - 1, None, None) for n in signal.shape)  # center after convolution; i.e. [m-1:, n-1:, ...]
	_ZERO = (0,) * signal.ndim

	# center data around mean. NOTE: numpy broadcasting rules make this work as expected
	if mask is None:
		signal = signal - np.mean(signal)
	else:
		n = np.sum(mask)  # number of (in bounds) samples. Counts the number of 1.0 elements.
		mean = np.sum(signal * mask) / n
		signal = (signal - mean) * mask

	# autocovariance achieved by convolution
	signal = fftconvolve(signal, signal[_REVERSED])

	# data is symetric around n - 1 in all dimensions, which correspond to delay/shift=0, since negative and positive shifts are equivalent
	signal = signal[_HALF]

	# signal[0, 0, ...] is the variance of each signal (i.e. column) since it corresponds to delay/shift=0
	with np.errstate(invalid="ignore", divide="ignore"):
		signal /= signal[_ZERO]

	return signal

def ndindex_as_array(shape) -> NDArray:
	""" Get all n-dimensional index combinations for an array with the given shape as a 2D NDArray with shape=(_, n). """
	_WILDCARD = -1
	return np.stack(np.indices(shape), axis=-1).reshape(_WILDCARD, len(shape))  # I hate python and numpy! There has to be an easier way to do this.
