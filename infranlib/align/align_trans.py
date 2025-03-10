import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

from align.matlab_cp2tform import get_similarity_transform_for_cv2


REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051], 
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.36550140],
    [62.72990036,  92.20410156]
]

# REFERENCE_FACIAL_POINTS = [[33.324059483, 56.865930561000006],
#  [72.08497925200001, 56.65153998900001],
#  [52.82771987900001, 78.910263058],
#  [36.904231265, 101.60205154],
#  [69.00289039600001, 101.42451171600001]]

DEFAULT_CROP_SIZE = (96, 112)
# DEFAULT_CROP_SIZE = (112, 112)

class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))

def get_reference_facial_points(output_size = None,
                                inner_padding_factor = 0.0,
                                outer_padding=(0, 0),
                                default_square = False):
    """
    Function:
    ----------
        get reference 5 key points according to crop settings:
        0. Set default crop_size:
            if default_square: 
                crop_size = (112, 112)
            else: 
                crop_size = (96, 112)
        1. Pad the crop_size by inner_padding_factor in each side;
        2. Resize crop_size into (output_size - outer_padding*2),
            pad into output_size with outer_padding;
        3. Output reference_5point;
    Parameters:
    ----------
        @output_size: (w, h) or None
            size of aligned face image
        @inner_padding_factor: (w_factor, h_factor)
            padding factor for inner (w, h)
        @outer_padding: (w_pad, h_pad)
            each row is a pair of coordinates (x, y)
        @default_square: True or False
            if True:
                default crop_size = (112, 112)
            else:
                default crop_size = (96, 112);
        !!! make sure, if output_size is not None:
                (output_size - outer_padding) 
                = some_scale * (default crop_size * (1.0 + inner_padding_factor))
    Returns:
    ----------
        @reference_5point: 5x2 np.array
            each row is a pair of transformed coordinates (x, y)
    """

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    # if (output_size and
    #         output_size[0] == tmp_crop_size[0] and
    #         output_size[1] == tmp_crop_size[1]):
    #     return tmp_5pts

    if (inner_padding_factor == 0 and
            outer_padding == (0, 0)):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

    # if not (0 <= inner_padding_factor <= 1.0):
    #     raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    # if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
    #         and output_size is None):
    #     output_size = tmp_crop_size * \
    #         (1 + inner_padding_factor * 2).astype(np.int32)
    #     output_size += np.array(outer_padding)

    # if not (outer_padding[0] < output_size[0]
    #         and outer_padding[1] < output_size[1]):
    #     raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
    #                             'and outer_padding[1] < output_size[1])')

    # if inner_padding_factor > 0:
    #     size_diff = tmp_crop_size * inner_padding_factor * 2
    #     tmp_5pts += size_diff / 2
    #     tmp_crop_size += np.round(size_diff).astype(np.int32)

    # size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2

    # if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
    #     raise FaceWarpException('Must have (output_size - outer_padding)'
    #                             '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    # scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    # tmp_5pts = tmp_5pts * scale_factor
    # tmp_crop_size = size_bf_outer_pad
    # reference_5point = tmp_5pts + np.array(outer_padding)
    # tmp_crop_size = output_size

    # return reference_5point

def get_affine_transform_matrix(src_pts, dst_pts):
    """
    Function:
    ----------
        get affine transform matrix 'tfm' from src_pts to dst_pts
    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points matrix, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points matrix, each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @tfm: 2x3 np.array
            transform matrix from src_pts to dst_pts
    """

    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm

def warp_and_crop_face(
    src_img,
    facial_pts,
    reference_pts = None,
    crop_size=(112, 112),
    align_type = 'smilarity'
    # align_type = 'affine'
    ):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

    # if reference_pts is None:
    #     if crop_size[0] == 96 and crop_size[1] == 112:
    #         reference_pts = REFERENCE_FACIAL_POINTS
    #     else:
    #         default_square = False
    #         inner_padding_factor = 0
    #         outer_padding = (0, 0)
    #         output_size = crop_size

    #         reference_pts = get_reference_facial_points(output_size,
    #                                                     inner_padding_factor,
    #                                                     outer_padding,
    #                                                     default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img