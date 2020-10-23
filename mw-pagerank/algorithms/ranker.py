import torch as th


# Possible edge strength functions - here an exponential edge strength was implemented
def expedge(x):
    return th.exp(x)


def logedge(x):
    return (ExpEdge(-x) + 1) ** (-1)


# Class performing Personalised PageRank given a set of weights
class Ranker:

    def __init__(self, nnodes, nfeats, weights, psi, device):

        self.nnodes = nnodes
        self.nfeats = nfeats
        self.weights = weights
        self.psi = psi
        self.device = device

    def strength_matrix(self):
        """
        Generate and compute edge strength matrix, given edge feature matrix and weights
        outputs:
            strength_m: torch.Tensor (|G| x |G| matrix)
                edge strength matrix
        """
        str_m = th.einsum('ijk,k->ij', self.psi, self.weights)
        strength_m = expedge(str_m)

        return strength_m

    def rw_matrix(self):
        """
        Generate and compute random walk transition matrix by row normalising edge strength matrix
        outputs:
            rw_m: torch.Tensor (|G| x |G| matrix)
                random walk transition matrix
        """
        strengths = self.strength_matrix()
        sum_strengths = strengths.sum(dim=1)
        sum_strengths = th.reshape(sum_strengths, (self.nnodes, 1))
        rw_m = th.div(strengths, sum_strengths)

        return rw_m

    def rwr_matrix(self, rw_m, source, alpha=0.3):
        """
        Allow for restarts in random walk process
        inputs:
            rw_m: torch.Tensor (|G| x |G| matrix)
                random walk transition matrix
            source: int
                index of source/restart node
            alpha: float (between 0 and 1)
                restart probability
        outputs:
            rwr_m: torch.Tensor (|G| x |G| matrix)
                random walk with restarts transition matrix
        """
        rwr_m = (1 - alpha) * rw_m
        rst = th.zeros(self.nnodes, self.nnodes, dtype=th.float, device=self.device)
        rst[:, source] += alpha
        rwr_m = th.add(rwr_m, rst)
        return rwr_m

    def rwr_iter(self, rw_m, source, eps=1e-5, maxIters=50):
        """
        Perform random walk with restart algorithm, commencing from source node
        inputs:
            rw_m: torch.Tensor (|G| x |G| matrix)
                random walk transition matrix
            source: int
                index of node list
            eps: float
                error tolerance of convergence
            maxIters: int
                maximum number of iterations allowed
        outputs:
            p: torch.Tensor (1 x |G| vector)
                unordererd ranking score vector for input nodes

        """

        rwr_matrix = self.rwr_matrix(rw_m, source)
        p = th.zeros(1, self.nnodes, dtype=th.float, requires_grad=False, device=self.device)
        p += 1 / self.nnodes
        for it in range(maxIters):
            new_p = th.mm(p, rwr_matrix)
            residual = th.norm(p - new_p)
            p = new_p
            if residual < eps:
                break

        return p