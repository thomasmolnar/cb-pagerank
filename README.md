# Implementation of  'Supervised Random Walk with Restarts' [Backstrom et al](https://arxiv.org/pdf/1011.4071.pdf) using Pytorch, to examine similarity rankings of Crunchbase company/investor data according to biased Spectrum list 

## __General information:__

* __Outline of project:__

The aim of this project was to implement the algorithm proposed by Backstrom et al using Pytorch and use it to analyse similarity rankings for knowledge graphs generated from Crunchbase entries. The advantageous use of [Pytorch](https://pytorch.org/docs/stable/index.html) enables GPU accelerated computations and automatic differentiation using [Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html), which ostensibly takes away the need of explicitly computing the gradient of optimisation functions for backpropagation.

* __Method:__

As outlined in the [paper](https://arxiv.org/pdf/1011.4071.pdf), the first step in the algorithm consists of conducting a **_random walk with restarts_** (equivalent to a **_personalised PageRank_**) starting from a designated source node, using predetermined weights (to be optimised). Following the ranking obtained, a cost function is computed depending on the position of the bias nodes with respect to other nodes in the ranking. This is done by adding amounts (calculated as a function of the bias and random node ranking) to the cost function as penalties for cases where bias nodes are ranked lower than random nodes. 

The same proccess is then repeated with each node, in the set, acting as the source node. This leads to an overall cost function, evaluated for the whole training set. From this function, the automatic differentiation package Autograd is utilised to compute the gradients of the cost function with respect to the weights. The optimised weights are then found by iterating through a gradient descent algorithm. 

* __How to use:__

In order to use the package, the training data input has to be in the form of a *__feature matrix__*. This entails a matrix with the same shape and sparsety as the adjacency matrix of the graph, but with feature vectors as elements. Furthermore, the algorithm is constructed such that the postive (bias) nodes are found to be at the start of the set of training nodes. The seperation between postive and negative nodes is then denoted by a *__limit index__*, which is initialised along with the feature matrix in the gradient descent class. 

Once the data has been preprocessed, the gradient descent method - grad_iterator - of the [gradient descent object](https://github.com/vada-oxford/gen-sim/blob/master/mw-pagerank/algorithms/grad_descent.py) can be called which will return the optimised weights. Uses of the package can be seen in the various cases below.  


## __[Synthetic Test Case](https://github.com/vada-oxford/gen-sim/blob/master/mw-pagerank/mw_pagerank_synthetic_test.ipynb):__

__1. Data:__ 

The full synthetic data set used in this test case was a randomly generated graph with 5000 nodes. Furthermore, edges between these nodes were randomly genereated according to a normal distribution. Associated with each edge was a random 'colour' attribute, i.e. the one-hot encoding of a colour. Four possible colours were considered: red, green, blue, yellow, orange. For example, a 'red' edge would have as feature vector [1, 0, 0, 0, 0]. 

From the full synthetic graph, a bias was included where all edges formed with a set of 250 bias nodes were 'red'. 250 bias nodes and 250 random nodes were then extracted from the graph, which constituted the training data. Using the nomenclature adopted in the paper, the bias nodes are the positive **_destination nodes_**, while the random nodes are the negative **_no-link nodes_**. 

__2. Results:__

Upon finding the optimised weights learnt through the training phase, a **_personalised PageRank_** of the whole synthetic graph was performed to test whether the bias nodes are found within the top portion of the ranking of all nodes. To visualise the results, a graph was plotted showing the number of bias nodes found within increasing subsets of the top ranked nodes. This graph is shown below: 

   ![Bias results](https://github.com/thomasmolnar/cb-pagerank/results/bias-synthethic-results.png)

The expected performance of the algorithm is illustrated by the saturation of all the bias nodes within the top ~500 nodes in the ranking. This shows the strength of the algorithm as a classifier according to the bias node attributes. 

As a control, the same ranking procedure was performed with randomly generated weights, which yielded the following graph:

   ![Random results](https://github.com/thomasmolnar/cb-pagerank/results/rand-synthethic-results.png)

The trend is clearly linear, showing the dispersion of the bias nodes throughout the ranking, with no clear clustering at the top.  

## __[Crunchbase Use Case](https://github.com/vada-oxford/gen-sim/blob/master/mw-pagerank/mw_pagerank_test.ipynb):__ 

__1. Data:__

The data analysed in this case consisted of companies listed on **Crunchbase**, with a multitude of attributes associated with them. In order to perform the algorithm, the attributes in question must be quantised in a vectorised format. This could be achieved easily with the following attributes: **_funding round_**, **_total funding in USD_**, **_employee count_**, hence these attributes were considered. These attributes were then normalised to have a mean of 0 and standard deviation of 1, in order to equalise the effect of each of them in the algorithm. 

When constructing the graph of only companies, edges were generated randomly and sparsely for each company node. This was done since the algorithm can act roughly as a classifier between bias and random nodes independent of graph structure (i.e. where edges were formed). The attributes associated with these edges were then simply the respective node attributes of the two nodes forming the edge.

__2. Results:__

Due to the nature of the algorithm being constituted of various nested loops, the runtime for large datasets becomes increasingly lengthy. Hence, to test the performance adequately, I only considered a set of 10000 companies (with the 600 Spectrum/bias companies included). From this set, as previously, a training set of 250 bias nodes and 250 random nodes was constructed and allowed for the optimised weights to be found. The following graph was found:

   ![Use case results](https://github.com/thomasmolnar/cb-pagerank/results/usecase-results.png)

Once again this graph illustrates the expected performance of the algorithm, placing the Spectrum list companies at the top of the ranking list. This is remarkable considering the small amount of features considered. The steep rise of the curve can be attributed to the fact that most of the Spectrum list nodes were sizeable well-established companies. This meant that the attributes associated to them, quantifying the scale of the company, were all large. In constrast, most of the randomly selected nodes are small companies, which have usually not surpassed many funding rounds. Hence the algorithm was easily able to find the optimised weights biased towards larger feature values. 

__3. Discussion & Possible Continuation:__

The results outlined were purely illustrative of the performance of the algorithm. However, in order to get concrete results further analysis is needed. For example. if we were to perform a personalised PageRank on the knowledge graph, neglecting to include the bias companies, we would be able to find companies with similarities to the Spectrum list, according to the attributes used. This shows the interesting link predicting aspects of the algorithm.  

With the outlined procedure, similarity rankings for various entities can be considered and a lot of room for experimentation is possible. I have only considered company nodes in this case in order to simplify the proceedings. However, various graphs could be constructed including differing nodes such as investors, industries and events, subject to the specifications of the task at hand. Note that in these cases, a suitable referencing system must be put in place in order to only consider the ranking of a certain type of nodes and not a mixture. Additionally, there is also room for experimentation in terms of the types of edges formed between the graph nodes. This, of course, will once again depend on the task at hand. 

One of the vital points of this task is the choice of attributes to include in the data. In the case above, I only considered explicitly quantitative attributes. However, if time would have permitted, I would have investigated word embedding in order represent certain attributes which are otherwise not vectorised. Possible extensions include word2vec, node2vec and deepwalk.

