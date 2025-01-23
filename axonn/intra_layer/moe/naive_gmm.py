# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

def gmm(a, b, batch_sizes, trans_a=False, trans_b=False):
    batch_sizes = batch_sizes.cpu().numpy()
    assert a.dim() == 2
    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        if b.dim() == 3:
            rhs = b[i, :, :].t() if trans_b else b[i, :, :]
            lhs = a[start:start + size, :].t() if trans_a else a[start:start + size, :]
        else:
            lhs = a[start:start + size, :].t() if trans_a else a[start:start + size, :]
            rhs = b[start:start + size, :].t() if trans_b else b[start:start + size, :]
        out.append(lhs @ rhs)
        start += size   
    return torch.cat(out)


