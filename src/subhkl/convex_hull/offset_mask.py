"""
OffsetMask data structure for efficiently storing small masks in a large
image.
"""
import numpy as np


class OffsetMask:
    def __init__(self, mask, offset):
        self.mask = mask
        self.offset = np.array(offset, dtype=int)

    def indices(self):
        return (
            slice(self.offset[0], self.offset[0] + self.mask.shape[0]),
            slice(self.offset[1], self.offset[1] + self.mask.shape[1])
        )

    def __iand__(self, other):
        # Probably the fastest implementation here is to create a new mask
        # array anyway, so we just fall back to __and__
        result = self & other
        self.mask = result.mask
        self.offset = result.offset

    def __ior__(self, other):
        result = self | other
        self.mask = result.mask
        self.offset = result.offset

    def __and__(self, other):
        # Result region is intersection of two mask regions
        result_min = np.maximum(self.offset, other.offset)
        result_max = np.minimum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )
        result_shape = result_max - result_min

        if np.any(result_shape < 0):
            return OffsetMask(np.zeros((0, 0), dtype=bool), (0, 0))

        my_i = (
            slice(result_min[0] - self.offset[0], result_max[0] - self.offset[0]),
            slice(result_min[1] - self.offset[1], result_max[1] - self.offset[1])
        )
        other_i = (
            slice(result_min[0] - other.offset[0], result_max[0] - other.offset[0]),
            slice(result_min[1] - other.offset[1], result_max[1] - other.offset[1])
        )
        result_mask = self.mask[my_i[0], my_i[1]] & other.mask[other_i[0], other_i[1]]

        return OffsetMask(result_mask, result_min)

    def __rand__(self, other):
        return self & other

    def __or__(self, other):
        # Result region is union of two mask regions
        result_min = np.minimum(self.offset, other.offset)
        result_max = np.maximum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )
        result_shape = result_max - result_min

        result_mask = np.zeros(result_shape, dtype=bool)
        my_offset_rel = self.offset - result_min
        my_i = (
            slice(my_offset_rel[0], my_offset_rel[0] + self.mask.shape[0]),
            slice(my_offset_rel[1], my_offset_rel[1] + self.mask.shape[1])
        )
        result_mask[my_i[0], my_i[1]] = self.mask

        other_offset_rel = other.offset - result_min
        other_i = (
            slice(other_offset_rel[0], other_offset_rel[0] + other.mask.shape[0]),
            slice(other_offset_rel[1], other_offset_rel[1] + other.mask.shape[1])
        )
        result_mask[other_i[0], other_i[1]] &= other.mask

        return OffsetMask(result_mask, result_min)

    def __ror__(self, other):
        return self | other

    def __invert__(self):
        return OffsetMask(~self.mask, self.offset)

    def __sub__(self, other):
        # self - other = self and not other
        # Find intersection
        int_min = np.maximum(self.offset, other.offset)
        int_max = np.minimum(
            self.offset + self.mask.shape,
            other.offset + other.mask.shape
        )

        # No need to do anything if intersection is empty
        if np.any(int_max - int_min) == 0:
            return OffsetMask(self.mask.copy(), self.offset)

        # Update mask in intersection
        my_i = (
            slice(int_min[0] - self.offset[0], int_max[0] - self.offset[0]),
            slice(int_min[1] - self.offset[1], int_max[1] - self.offset[1])
        )
        other_i = (
            slice(int_min[0] - other.offset[0], int_max[0] - other.offset[0]),
            slice(int_min[1] - other.offset[1], int_max[1] - other.offset[1])
        )
        # Copy outside of intersection because ~other is True in my region
        # (set) minus other region
        result_mask = self.mask.copy()

        # and not other in the intersection
        result_mask[my_i[0], my_i[1]] &= ~other.mask[other_i[0], other_i[1]]

        return OffsetMask(result_mask, self.offset)

    def full(self, shape):
        full_result = np.zeros(shape, dtype=bool)
        full_result[
            self.offset[0]: self.offset[0] + self.mask.shape[0],
            self.offset[1]: self.offset[1] + self.mask.shape[1]
        ] = self.mask

        return full_result

    def nonzero(self):
        x, y = np.nonzero(self.mask)
        x += self.offset[0]
        y += self.offset[1]
        return x, y
