# Block Sparsity on CPU

A very different take on openai block sparsity for GPU

A fully connected network is trained with mnist 
Using block sparsity.
Sparsity is used to reduce CPU time in both backward and
Forward phases of training.

Mnist can be trained with less than 2 percent loss
In less than 12 seconds on a single thread and
Less than 3 seconds using multiple threads.

For comparison mlpack gets worse accuracy using more or less the same architecture in 50 seconds on a single thread on the same hardware 

This is achieved in the following ways

* Non 0 matrix values are extracted and re-ordered
* Cache optimized routines for matrix vector multiply during backprop using blocks instead of individual values
* SIMD routines for blockwise dot product and scalar vector multiplication are added
* Block sparsity is gradually added during the first few epochs which then also reduces forward multiplies

Note: the 'main' branch is currently being developed

# Build and Run

To build on replit and other *nix's

* (open console/shell)
* git submodule init
* git submodule update
* mkdir bld
* cd bld
* cmake ..
* make
* ./nnnn

output like this is expected

```
...
[INF][2023-03-04 14:44:25][stochastic_gradient_descent] Completing Epoch 16 . complete  100 %
[INF][2023-03-04 14:44:25][stochastic_gradient_descent] estimated acc.:  0.931667
[INF][2023-03-04 14:44:25][stochastic_gradient_descent] last epoch duration: 0.566954 s, lr: 0.007835
[INF][2023-03-04 14:44:26][stochastic_gradient_descent] Best Epoch 9
...
```

You can then change some of the parameters in the 'models/mnist_fc.json'
it's also easy to make changes to the sources itself


