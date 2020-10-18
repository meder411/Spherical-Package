import torch
import cv2
import math

from .conversions import *
from .io import torch2numpy, numpy2torch
from .tangent_images import get_valid_coordinates, convert_tangent_image_coordinates_to_spherical
import _spherical_distortion_ext._mesh as _mesh


def compute_sift_keypoints(img,
                           nfeatures=0,
                           nOctaveLayers=3,
                           contrastThreshold=0.04,
                           edgeThreshold=10,
                           sigma=1.6):
    """
    Expects 3 x H x W torch tensor

    Returns [M x 4 (x, y, s, o), M x 128]
    """
    # Convert to numpy and ensure it's a uint8
    img = torch2numpy(img.byte())

    # Initialize OpenCV SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers,
                                       contrastThreshold, edgeThreshold, sigma)

    # Keypoints is a list of lenght N, desc is N x 128
    keypoints, desc = sift.detectAndCompute(img, None)
    if len(keypoints) > 0:
        coords = torch.tensor([kp.pt for kp in keypoints])
        orientation = torch.tensor(
            [kp.angle * math.pi / 180 for kp in keypoints])
        scale = torch.tensor([kp.size for kp in keypoints])
        desc = torch.from_numpy(desc)
        return torch.cat(
            (coords, scale.unsqueeze(1), orientation.unsqueeze(1)), -1), desc
    return None


def draw_keypoints(img, keypoints):
    """
    Visualize keypoints
    img: 3 x H x W
    keypoints: N x 4
    """
    kp = [
        cv2.KeyPoint(k[0], k[1], k[2], math.degrees(k[3])) for k in keypoints
    ]
    out_img = cv2.drawKeypoints(
        torch2numpy(img),
        kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return numpy2torch(out_img)


def compute_crop(image_shape, crop_degree=0):
    """Compute padding space in an equirectangular images"""
    crop_h = 0
    if crop_degree > 0:
        crop_h = image_shape[0] // (180 / crop_degree)

    return crop_h


def sift_equirectangular(img, crop_degree=0):
    """
    img: torch style (C x H x W) torch tensor
    crop_degree: [optional] scalar value in degrees dictating how much of input equirectangular image is 0-padding

    returns [erp_kp, erp_desc] (M x 4, M x 128)
    """

    # ----------------------------------------------
    # Compute SIFT descriptors on equirect image
    # ----------------------------------------------
    erp_kp_details = compute_sift_keypoints(img)
    erp_kp = erp_kp_details[0]
    erp_desc = erp_kp_details[1]

    # If top top and bottom of image is padding
    crop_h = compute_crop(img.shape[-2:], crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (erp_kp[:, 1] > crop_h) & (erp_kp[:, 1] < img.shape[1] - crop_h)
    erp_kp = erp_kp[mask]
    erp_desc = erp_desc[mask]

    return erp_kp, erp_desc


def sift_tangent_images(tan_img,
                        base_order,
                        sample_order,
                        image_shape,
                        crop_degree=0):
    """
    Extracts only the visible SIFT features from a collection of tangent images and transfers them to coordinates on the equirectangular image. That is, only returns the keypoints visible to a spherical camera at the center of the icosahedron.

    tan_img: 3 x N x H x W
    corners: N x 4 x 3 coordinates of tangent image corners in 3D
    image_shape: (H, W) of equirectangular image that we render back to
    crop_degree: [optional] scalar value in degrees dictating how much of input equirectangular image is 0-padding

    returns [visible_kp, visible_desc] (M x 4, M x 128) all keypoint coordinates are on the equirectangular image
    """
    # ----------------------------------------------
    # Compute SIFT descriptors for each patch
    # ----------------------------------------------
    kp_list = []  # Stores keypoint coords
    desc_list = []  # Stores keypoint descriptors
    quad_idx_list = []  # Stores quad index for each keypoint
    for i in range(tan_img.shape[1]):
        kp_details = compute_sift_keypoints(tan_img[:, i, ...])

        if kp_details is not None:
            # Compute visible keypoints
            valid_mask = get_valid_coordinates(base_order,
                                               sample_order,
                                               i,
                                               kp_details[0][:, :2],
                                               return_mask=True)[1]
            visible_kp = kp_details[0][valid_mask]
            visible_desc = kp_details[1][valid_mask]

            # Convert tangent image coordinates to equirectangular
            visible_kp[:, :2] = convert_spherical_to_image(
                torch.stack(
                    convert_tangent_image_coordinates_to_spherical(
                        base_order, sample_order, i, visible_kp[:, :2]), -1),
                image_shape)

            kp_list.append(visible_kp)
            desc_list.append(visible_desc)

    # Assemble keypoint data
    all_visible_kp = torch.cat(kp_list, 0).float()  # M x 4 (x, y, s, o)
    all_visible_desc = torch.cat(desc_list, 0).float()  # M x 128

    # If top top and bottom of image is padding
    crop_h = compute_crop(image_shape, crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (all_visible_kp[:, 1] > crop_h) & (all_visible_kp[:, 1] <
                                              image_shape[0] - crop_h)
    all_visible_kp = all_visible_kp[mask]  # M x 4
    all_visible_desc = all_visible_desc[mask]  # M x 128
    return all_visible_kp, all_visible_desc