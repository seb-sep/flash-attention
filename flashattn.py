import torch
import torch.nn.functional as F
import math


M = 16


def flashattn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert Q.shape == K.shape == V.shape
    assert len(Q.shape) == 2

    K = K.T

    # seq length, inner dim of representations
    N, d = Q.shape

    # of kv vectors per tile
    bc = math.ceil(M/(4*d))
    tc = math.ceil(N/bc)
    K_shared = torch.empty((d, bc), dtype=Q.dtype)
    V_shared = torch.empty((bc, d), dtype=Q.dtype)

    # of q vectors per tile
    br = min(math.ceil(M/(4*d)), d)
    tr = math.ceil(N/br)
    Q_shared = torch.empty((br, d), dtype=Q.dtype)

    # output matrix
    O = torch.zeros_like(Q)
    O_shared = torch.empty_like(Q_shared)

    # intermediate rowmaxes
    m = torch.full((N,), -torch.inf)
    m_shared = torch.empty(br, dtype=Q.dtype)
    # intermediate normalization constants
    l = torch.full((N,), 0)
    l_shared = torch.empty(br, dtype=Q.dtype)

    for i in range(tc):
        # load k, v chunks
        # make sure we load in k as its transposed version
        K_shared[:, :] = K[:, i*bc:(i+1)*bc]
        V_shared[:, :] = V[i*bc:(i+1)*bc, :]

        # print(f'K: {K_shared}\n, V: {V_shared}')
        for j in range(tr):
            # load in q, o, m, l
            Q_shared[:, :] = Q[j*br:(j+1)*br, :]
            # if i == 0: print(f'Q: {Q_shared}')
            O_shared[:, :] = O[j*br:(j+1)*br, :]
            m_shared[:] = m[j*br:(j+1)*br]
            l_shared[:] = l[j*br:(j+1)*br]

            S = Q_shared @ K_shared

            # get row-wise softmax statistics
            mt = S.max(dim=1).values
            S_shift = S - mt.reshape(-1, 1)
            Pt = torch.exp(S_shift)
            lt = Pt.sum(dim=1)

            # compute new statistics
            m_new = torch.max(mt.flatten(), m_shared)
            # l_new = torch.exp((m_shared - m_new) + l_shared.log()) + torch.exp((mt - m_new) + lt.log())
            l_new = (torch.exp(m_shared - m_new) * l_shared) + (torch.exp(mt - m_new) * lt)


            # update chunk of output
            O[j*br:(j+1)*br, :] = (l_shared * torch.exp(m_shared - m_new) * O_shared + torch.exp(mt - m_new) * Pt @ V_shared) / l_new
            
            m[j*br:(j+1)*br] = m_new[:]
            l[j*br:(j+1)*br] = l_new[:]

    return O

def dumb_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """equivalent to F.scaled_dot_product_attention(Q, K, V, scale=1)"""
    return torch.softmax(Q @ K.T, dim=1) @ V

Q = torch.rand((2, 2))
K = torch.rand((2, 2))
V = torch.rand((2, 2))

print(dumb_attn(Q, K, V), flashattn(Q, K, V))

