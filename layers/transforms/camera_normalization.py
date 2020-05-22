import torch
import math

from ..functional import unresample
from ..layer_utils import InterpolationType


class CameraNormalization(object):
    """
    Resamples an image to a different intrinsic matrix
    """

    def __init__(self,
                 out_fov,
                 out_shape,
                 random_shift=False,
                 interpolation=InterpolationType.BILINEAR):
        """
        out_fov: (fov_y, fov_x) in degrees
        out_shape: (H, W) in pixels
        """
        assert type(out_fov) in [tuple, list] and len(out_fov) == 2, \
            'out_fov should be a tuple or list as (fov_y, fov_x)'
        assert type(out_shape) in [tuple, list] and len(out_shape) == 2, \
            'out_shape should be a tuple or list as (H, W)'

        # Store output dimensions
        self.H_out, self.W_out = out_shape  # pixels
        self.fovy_out, self.fovx_out = out_fov  # degrees

        # Store whether or not to apply random shifts
        self.random_shift = random_shift

        # Store the interpolation type
        self.interpolation = interpolation

        # Create the output grid
        y, x = torch.meshgrid(
            torch.arange(self.H_out), torch.arange(self.W_out))
        self.grid = torch.stack(
            (x.float(), y.float(), torch.ones(self.H_out, self.W_out)), -1)

        # Compute the new camera matrix
        self.fx = self.W_out / (2 * math.tan(math.radians(self.fovx_out) / 2))
        self.fy = self.H_out / (2 * math.tan(math.radians(self.fovy_out) / 2))
        self.cx = self.W_out / 2
        self.cy = self.H_out / 2
        self.new_Kinv_T = torch.tensor(
            [[1 / self.fx, 0.0, 0.0], [0.0, 1.0 / self.fy, 0.0],
             [-self.cx / self.fx, -self.cy / self.fy, 1.0]])

    def get_K(self):
        return torch.tensor([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                             [0.0, 0.0, 1.0]])

    def make_sample_grid(self, K_in):
        # Returns a dim x dim x 2 sample grid

        return (self.grid.view(-1, 3) @ self.new_Kinv_T @ K_in.T).view(
            self.W_out, self.H_out, 3)[..., :2].contiguous()

    def compute_random_shift(self, shape_in, K_in):
        H_in, W_in = shape_in
        fx_in = K_in[0, 0]
        fy_in = K_in[1, 1]
        cx_in = K_in[0, 2]
        cy_in = K_in[1, 2]

        # Compute the minimum deltas
        min_dx = fx_in * self.cx / self.fx - cx_in
        min_dy = fy_in * self.cy / self.fy - cy_in

        # Compute the maximum deltas
        max_dx = W_in - cx_in - (fx_in / self.fx) * (self.W_out - self.cx)
        max_dy = H_in - cy_in - (fy_in / self.fy) * (self.H_out - self.cy)

        # Compute the random shift
        dx = (torch.rand(1) * (max_dx - min_dx) + min_dx).item()
        dy = (torch.rand(1) * (max_dy - min_dy) + min_dy).item()

        return dx, dy

    def __call__(self, img, K_in, shift=None):
        """
        img: B x C x H x W image
        K_in: 3 x 3 intrinsics matrix
        shift: [optional] (dx, dy) that overrides the internal random shift setting 
        """

        # Default, no shift
        dx = dy = 0

        # If desired, randomly shift the camera's principal point to adjust where the image is cropped
        if self.random_shift and shift == None:
            dx, dy = self.compute_random_shift(img.shape[-2:], K_in)

        # If a shift is provided, use that shift
        if shift is not None:
            assert len(shift) == 2, 'shift parameter should be (dx, dy)'
            dx, dy = shift

        K_in = K_in.clone()  # Don't modify the original matrix, just in case
        K_in[0, 2] += dx
        K_in[1, 2] += dy

        # Create a sample map to unresample to
        sample_grid = self.make_sample_grid(K_in)

        # Add the batch dimension if necessary for the resample call
        added_batch_dim = False
        if img.dim() == 3:
            img.unsqueeze_(0)
            added_batch_dim = True

        # Resample the input according to the sample grid
        output = unresample(img, sample_grid, self.interpolation)

        # Remove the batch dimension if it wasn't originally there
        if added_batch_dim:
            output.squeeze_(0)

        return output