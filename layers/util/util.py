import torch
import time


def time_cuda(func, param_list):
    '''
    Times the GPU runtime of a function, returning the output and wall-clock time.
    '''
    torch.cuda.synchronize()
    s = time.time()
    output = func(*param_list)
    torch.cuda.synchronize()
    t = time.time() - s
    return output, t


def batched_index_select(input, dim, index):
    '''
    input: B x * x ... x *
    dim: scalar
    index: B x M
    '''
    dim %= input.dim()
    if index.dim() == 2:
        views = [input.shape[0]] + \
         [1 if i != dim else -1 for i in range(1, input.dim())]
    elif index.dim() == 1:
        views = [1 if i != dim else -1 for i in range(input.dim())]
    else:
        assert False, 'index must have 1 or 2 dimensions ({})'.format(
            index.dim())
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def batched_scatter(input, dim, index):
    '''
    input: B x * x ... x *
    dim: scalar
    index: M
    '''
    dim %= input.dim()
    views = [1 if i != dim else -1 for i in range(input.dim())]
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    z = torch.zeros_like(input)
    return z.scatter_(dim, index, input)


def normals2rgb(normals):
    return ((normals * 127.5) + 127.5).byte()


def visualize_rgb(rgb, mu=0.0, sigma=1.0):
    # Scale back to [0,255]
    rgb = rgb * sigma + mu
    return (255 * rgb).byte()


def visualize_mask(mask):
    '''Visualize the data mask'''
    mask /= mask.max()
    return (255 * mask).byte()


def points_in_triangle_2d(pts, triangle):
    """
    pts: * x 2
    triangle: 3 x 2

    Returns a binary mask of which points fall inside the 2D triangle
    """

    def sign(pts, v0, v1):
        return (pts[..., 0] - v1[0]) * (v0[1] - v1[1]) - (v0[0] - v1[0]) * (
            pts[..., 1] - v1[1])

    d1 = sign(pts, triangle[0, :], triangle[1, :])
    d2 = sign(pts, triangle[1, :], triangle[2, :])
    d3 = sign(pts, triangle[2, :], triangle[0, :])
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)
