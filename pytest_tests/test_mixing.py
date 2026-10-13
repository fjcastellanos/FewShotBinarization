import random
import torch

from modern_fewshot.mixing import cutmix_quadrants, mixup_batch, mixup_partial_loss


def test_cutmix_single_sample_preserves_aligned_labels_and_validity():
    x = torch.arange(16).float().reshape(1,1,4,4)
    y = x[:,0].long()
    v = torch.zeros(1,4,4,dtype=torch.bool)
    v[0,0,0] = True
    xo,yo,vo = cutmix_quadrants(x,y,v,random.Random(3))
    assert torch.equal(xo[:,0].long(),yo)
    assert vo.sum() == v.sum()
    assert not torch.equal(x,xo)


def test_cutmix_multi_sample_and_odd_dimensions():
    x = torch.arange(2*1*5*7).float().reshape(2,1,5,7)
    y = x[:,0].long()
    v = torch.ones(2,5,7,dtype=torch.bool)
    xo,yo,vo = cutmix_quadrants(x,y,v,random.Random(5))
    assert xo.shape == x.shape
    assert torch.equal(xo[:,0].long(),yo)
    assert vo.all()


def test_mixup_preserves_partial_supervision_and_backprop():
    x = torch.stack([torch.zeros(1,4,4),torch.ones(1,4,4)])
    y = torch.zeros(2,4,4,dtype=torch.long)
    y[1]=1
    v = torch.zeros(2,4,4,dtype=torch.bool)
    v[:,1,1]=True
    mixed,ta,va,tb,vb,lam = mixup_batch(x,y,v,random.Random(2))
    assert torch.all((lam>=0.25)&(lam<=0.75))
    assert not torch.equal(mixed,x)
    logits = torch.zeros(2,2,4,4,requires_grad=True)
    loss = mixup_partial_loss(logits,ta,va,tb,vb,lam)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:,:,0,0]) == 0


def test_mixup_one_batch_is_safe_noop():
    x=torch.rand(1,3,4,4)
    y=torch.zeros(1,4,4,dtype=torch.long)
    v=torch.ones(1,4,4,dtype=torch.bool)
    assert torch.equal(mixup_batch(x,y,v,random.Random(1))[0],x)
