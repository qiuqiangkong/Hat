import numpy as np

class BaseTransformation(object):
    def __init__(self):
        pass
        
    """
    xs: list of ndarray
    ys: list of ndarray
    """
    def transform(self, xs, ys):
        raise NotImplementedError("transform() need to be implemented!")

class ImageAugmentation(BaseTransformation):
    """Image augmentation. The batch of images is augmentated randomly. 
    
    Args:
      shift: int
      flip: bool
    """
    def __init__(self, shift=0, flip=False):
        self._shift_ = shift
        self._flip_ = flip
        
    """
    xs: list of ndarray
    ys: list of ndarray
    """
    def transform(self, xs, ys):
        assert len(xs) == 1, "ImageAugmentation only support single input!"
        x = xs[0]
        assert (x.ndim == 3 or x.ndim == 4), "Input must have dimension 3 or 4"
        data_type = x.dtype
        x_shape = x.shape
        x_ndim = x.ndim
        if x_ndim == 3: x = x.reshape((x_shape[0], 1, x_shape[1], x_shape[2]))
        if self._shift_: x = self.shift(x)
        if self._flip_: x = self.flip(x)
        if x_ndim == 3: x = x.reshape(x_shape)
        # xs = [x.astype(data_type)]
        xs = [x]
        return xs, ys
        
    def shift(self, x):
        shift = self._shift_
        (n_samples, n_channels, height, width) = x.shape
        x_pad = np.zeros((n_samples, n_channels, height+shift*2, width+shift*2))
        x_pad[:, :, shift:height+shift, shift:width+shift] = x
        rand_shift = np.random.randint(low=0, high=shift*2, size=(n_samples, 2))
        x_new = np.zeros_like(x)
        for n in xrange(n_samples):
            [row_bgn, col_bgn] = rand_shift[n]
            x_new[n] = x_pad[n, :, row_bgn:row_bgn+height, col_bgn:col_bgn+width]
            
        return x_new
        
    def flip(self, x):
        n_samples = x.shape[0]
        binary = np.random.randint(low=0, high=2, size=n_samples)
        flip_x = x[...,::-1]
        x_new = x * binary[:,None,None,None] + flip_x * (1. - binary)[:,None,None,None]
        return x_new