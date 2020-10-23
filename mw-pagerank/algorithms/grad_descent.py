from algorithms.ranker import Ranker
import torch as th
import time

# Possible loss functions - here WMW loss was found to work the best
def squaredloss(x, b):
    return max((x+b)**2, 0)


def wmw_loss(x, b):
    return 1./(1.+th.exp(-x/b))

# Class performing gradient descent
class GradDescent:
    
    def __init__(self, psi, device, limit_ind):
        self.psi = psi
        self.device = device
        self.limit_ind = limit_ind
        self.nnodes = len(self.psi[0])
        self.nfeats = len(self.psi[0,0])
        
    def opt_function(self, weights, lamb=1., b=1e-4):
        """
        Compute 'F(w)', function to be optimised 
        inputs:
            weights: torch.Tensor(|PSI| x 1)
                parameters of funciton to be used to compute loss function 
            b: float
                WMW loss margin
            lamb: float (between 0 and 1)
                regularization parameter
        outputs:
            f_w: float
                cost function evaluated for whole candidate graph for specific parameters [w]  

        """
            
        graph_ranker = Ranker(self.nnodes, self.nfeats, weights, self.psi, self.device)
        graph_rw_matrix = graph_ranker.rw_matrix() # Generate random walk matrix only once for whole data set
        source_list = range(self.nnodes)

        #Compute cost function for the whole training data set, by iteratively adding loss for each candidate data set
        tot = th.tensor([0], dtype=th.float, device=self.device)
        for i in source_list:
            p_scores = graph_ranker.rwr_iter(graph_rw_matrix, i)
            p_scores = th.squeeze(p_scores)
            p_d = p_scores[:self.limit_ind]
            p_l = p_scores[self.limit_ind:]
            for p in p_d:
                diff = p_l - p
                loss = wmw_loss(diff, b)
                tot += th.sum(diff)
        
        f_w = th.norm(weights) + lamb*tot

        return f_w
                       
    def grad_iterator(self, learn_rate=1e-3, gamma=0.9, maxIters=100):
        """
        Iterate through gradient descent method to obtain optimized weights
        inputs:
            learn_rate: float
                learning rate for optimiser
            gamma: float (between 0 and 1)
                Nesterov momentum parameter for optimiser
            maxIters: int
                maximum number of iterations allowed
        outputs:
            weights: torch.Tensor (1 x |G| vector)
                optimized weights
        """
        weights = th.zeros(self.nfeats, dtype=th.float, requires_grad=True, device=self.device)
        optimizer = th.optim.SGD([weights], lr=learn_rate, momentum=gamma)
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

        epoch = 0
        for i in range(maxIters):
            start = time.time()
            optimizer.zero_grad()
            f_w = self.opt_function(weights)
            f_w.backward()
            end = time.time()
            length = end - start
            if epoch%10==0:
                print('Epoch={}, f_w={}, weights={}, grad={} ({}s)'.format(epoch, f_w, weights, weights.grad, length))
            if th.isnan(f_w).any()==1 or th.isnan(weights).any()==1:
                print('Loss function and gradients computed is nan')
                break
            if th.isinf(f_w).any()==1 or th.isinf(weights).any()==1:
                print('Loss function and gradients computed is inf')
                break
            optimizer.step()
            scheduler.step()
            epoch+=1
        
        return weights