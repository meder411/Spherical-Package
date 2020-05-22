import torch


def iou_score(pred_cls, true_cls, nclass, drop=(), mask=None):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    assert pred_cls.shape == true_cls.shape, 'Shape of predictions should match GT'
    if mask is not None:
        assert mask.dim() == true_cls.dim(), \
            'Mask should have the same dimensions as inputs'
    intersect_ = torch.zeros(nclass - len(drop), device=pred_cls.get_device())
    union_ = torch.zeros(nclass - len(drop), device=pred_cls.get_device())
    idx = 0
    for i in range(nclass):
        if i not in drop:
            intersect = (pred_cls == i).byte() + (true_cls == i).byte()
            if mask is not None:
                intersect *= mask.byte()
            intersect = intersect.eq(2).sum()
            union = (pred_cls == i).byte() + (true_cls == i).byte()
            if mask is not None:
                union *= mask.byte()
            union = union.ge(1).sum()
            intersect_[idx] = intersect
            union_[idx] = union
            idx += 1
    return intersect_, union_


def accuracy(pred_cls, true_cls, nclass, drop=(), mask=None):

    assert pred_cls.shape == true_cls.shape, 'Shape of predictions should match GT'
    if mask is not None:
        assert mask.dim() == true_cls.dim(), \
            'Mask should have the same dimensions as inputs'
    if mask is None:
        positive = torch.histc(
            true_cls.cpu().float(), bins=nclass, min=0, max=nclass)
    else:
        positive = torch.histc(
            true_cls[mask.expand_as(true_cls)].cpu().float(),
            bins=nclass,
            min=0,
            max=nclass)
    positive = positive.to(pred_cls.get_device())
    per_cls_counts = torch.zeros(
        nclass - len(drop), device=pred_cls.get_device())
    tpos = torch.zeros(nclass - len(drop), device=pred_cls.get_device())
    idx = 0
    for i in range(nclass):
        if i not in drop:
            true_positive = (pred_cls == i).byte() + (true_cls == i).byte()
            if mask is not None:
                true_positive *= mask.byte()
            true_positive = true_positive.eq(2).sum()
            tpos[idx] = true_positive
            per_cls_counts[idx] = positive[i]
            idx += 1
    return tpos, per_cls_counts