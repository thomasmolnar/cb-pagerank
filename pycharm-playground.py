import torch as th
import matplotlib.pyplot as plt
import random
from algorithms.grad_descent import *
from algorithms.ranker import *

device = 'cuda' if th.cuda.is_available() else 'cpu'

#Random adjacency matrix
nnodes = 100
rand_m = th.empty(nnodes, nnodes, dtype=th.float, device=device).uniform_(0,1)
adj_m = th.bernoulli(rand_m)

#Create random feature matrix with features as 'colours' psi = [red, green, blue, yellow, orange]
nfeats = 5
feat_matrix = th.empty(nnodes, nnodes, nfeats, dtype=th.float, device=device).uniform_(0,1)
feat_matrix = th.bernoulli(feat_matrix)

# Biased synthetic training data set for first ntrain nodes
ntrain = int(10)
train_adj = th.narrow(adj_m, 0, 0, ntrain)
train_adj = th.narrow(train_adj, 1, 0, ntrain)

train_feat = th.narrow(feat_matrix, 0, 0, ntrain)
train_feat = th.narrow(train_feat, 1, 0, ntrain)
train_feat[0:ntrain // 2, :, :] = th.tensor([1, 0, 0, 0, 0], dtype=th.float, device=device)
# create random destination/no-link nodes
pd_ind, pl_ind = range(0, ntrain // 2), range(ntrain // 2, ntrain)

# Optimised weights for training set
print('Loading optimised weights...')
rand_descent = GradDescent(train_adj, train_feat, device, pd_ind, pl_ind)
opt_weights = rand_descent.grad_iterator()
print('Optimised weights are {}'.format(opt_weights))

# Full personalised PageRank on whole data set using random training data
print('Loading graph PageRank...')
source_node = random.randint(0, nnodes)
full_ranker = Ranker(nnodes, nfeats, opt_weights, adj_m, feat_matrix, device)
p_scores, residuals, iters = full_ranker.rwr_iter(source_node)
p_scores = th.transpose(p_scores, 0, 1)

# Order PR scores
scores_i = {i: p_scores[i] for i in range(nnodes)}
new_ranking = sorted(scores_i, key=scores_i.get, reverse=True)

# Check graph of bias nodes
step = 0
steps = []
intersection = []
bias_nodes = set(range(ntrain // 2))
while step < nnodes:
    steps.append(step)
    numb = len(set(new_ranking[:step]).intersection(bias_nodes))
    intersection.append(numb)
    step += 5

# Graph time
fig, axes = plt.subplots(1, 2)

axes[0].plot(steps, intersection)
axes[1].plot(steps, intersection)

axes[0].legend(loc='lower right')
axes[1].legend(loc='upper left')
axes[1].set_xscale('log')
plt.show()
