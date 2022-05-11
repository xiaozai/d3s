import torch
from pytracking.features.preprocessing import sample_patch
from pytracking import TensorList

import cv2
import torch.nn.functional as F

class ExtractorBase:
    """Base feature extractor class.
    args:
        features: List of features.
    """
    def __init__(self, features):
        self.features = features

    def initialize(self):
        for f in self.features:
            f.initialize()

    def free_memory(self):
        for f in self.features:
            f.free_memory()


class SingleResolutionExtractor(ExtractorBase):
    """Single resolution feature extractor.
    args:
        features: List of features.
    """
    def __init__(self, features):
        super().__init__(features)

        self.feature_stride = self.features[0].stride()
        if isinstance(self.feature_stride, (list, TensorList)):
            self.feature_stride = self.feature_stride[0]

    def stride(self):
        return self.feature_stride

    def size(self, input_sz):
        return input_sz // self.stride()

    def extract(self, im, pos, scales, image_sz):
        if isinstance(scales, (int, float)):
            scales = [scales]

        # Get image patches
        im_patches = torch.cat([sample_patch(im, pos, s*image_sz, image_sz) for s in scales])

        # Compute features
        feature_map = torch.cat(TensorList([f.get_feature(im_patches) for f in self.features]).unroll(), dim=1)

        return feature_map


class MultiResolutionExtractor(ExtractorBase):
    """Multi-resolution feature extractor.
    args:
        features: List of features.
    """
    def __init__(self, features):
        super().__init__(features)
        self.is_color = None

    def stride(self):
        return torch.Tensor(TensorList([f.stride() for f in self.features if self._return_feature(f)]).unroll())

    def size(self, input_sz):
        return TensorList([f.size(input_sz) for f in self.features if self._return_feature(f)]).unroll()

    def dim(self):
        return TensorList([f.dim() for f in self.features if self._return_feature(f)]).unroll()

    def get_fparams(self, name: str = None):
        if name is None:
            return [f.fparams for f in self.features if self._return_feature(f)]
        return TensorList([getattr(f.fparams, name) for f in self.features if self._return_feature(f)]).unroll()

    def get_attribute(self, name: str, ignore_missing: bool = False):
        if ignore_missing:
            return TensorList([getattr(f, name) for f in self.features if self._return_feature(f) and hasattr(f, name)])
        else:
            return TensorList([getattr(f, name, None) for f in self.features if self._return_feature(f)])

    def get_unique_attribute(self, name: str):
        feat = None
        for f in self.features:
            if self._return_feature(f) and hasattr(f, name):
                if feat is not None:
                    raise RuntimeError('The attribute was not unique.')
                feat = f
        if feat is None:
            raise RuntimeError('The attribute did not exist')
        return getattr(feat, name)

    def _return_feature(self, f):
        return self.is_color is None or self.is_color and f.use_for_color or not self.is_color and f.use_for_gray

    def set_is_color(self, is_color: bool):
        self.is_color = is_color

    def extract(self, im, pos, scales, image_sz, dp=None, raw_depth=None):
        """Extract features.
        args:
            im: Image.
            dp: Depth
            pos: Center position for extraction.
            scales: Image scales to extract features from.
            image_sz: Size to resize the image samples to before extraction.
        """
        if isinstance(scales, (int, float)):
            scales = [scales]
        # Get image patches
        im_patches = torch.cat([sample_patch(im, pos, s*image_sz, image_sz) for s in scales])
        # Compute features
        feature_map = TensorList([f.get_feature(im_patches) for f in self.features]).unroll()

        dp_patches = None
        if dp is not None:
            dp_patches = torch.cat([sample_patch(dp, pos, s*image_sz, image_sz) for s in scales])
            dp_patches = TensorList([dp_patches for _ in self.features]).unroll()

        raw_dp_patches = None
        if raw_depth is not None:
            raw_dp_patches = torch.cat([sample_patch(raw_depth, pos, s*image_sz, image_sz) for s in scales])
            raw_dp_patches = TensorList([raw_dp_patches for _ in self.features]).unroll()

        if dp is not None and raw_depth is not None:
            return feature_map, dp_patches, im_patches, raw_dp_patches
        elif dp is not None and raw_depth is None:
            return feature_map, dp_patches, im_patches
        elif dp is None and raw_depth is not None:
            return feature_map, raw_dp_patches


    def extract_transformed(self, im, pos, scale, image_sz, transforms, dp=None, raw_depth=None):
        """Extract features from a set of transformed image samples.
        args:
            im: Image.
            dp: Depth
            pos: Center position for extraction.
            scale: Image scale to extract features from.
            image_sz: Size to resize the image samples to before extraction.
            transforms: A set of image transforms to apply.
        """
        # Get image patche
        im_patch = sample_patch(im, pos, scale*image_sz, image_sz) # [1, 3, 512, 512]
        # Apply transforms
        im_patches = torch.cat([T(im_patch) for T in transforms]) # [27, 3, 256, 256]
        # Compute features
        feature_map = TensorList([f.get_feature(im_patches) for f in self.features]).unroll()

        dp_patches = None
        if dp is not None:
            dp_patch = sample_patch(dp, pos, scale*image_sz, image_sz) # [1, 1, 512, 512]
            dp_patches = torch.cat([T(dp_patch) for T in transforms])
            dp_patches = TensorList([dp_patches for f in self.features]).unroll() # [27, 1, 256, 256]

        raw_d_patches = None
        if raw_depth is not None:
            raw_d_patch = sample_patch(raw_depth, pos, scale*image_sz, image_sz)
            raw_d_patches = torch.cat([T(raw_d_patch) for T in transforms])
            raw_d_patches = TensorList([raw_d_patches for f in self.features]).unroll()

        if dp is not None and raw_depth is not None:
            return feature_map, im_patches, dp_patches, raw_d_patches
        elif dp is not None and raw_depth is None:
            return feature_map, im_patches, dp_patches
        elif dp is None and raw_depth is not None:
            return feature_map, raw_d_patches
