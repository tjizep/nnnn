# Block Sparsity

A fully connected network is trained with mnist 
Using block sparsity.
Sparsity is used to reduce CPU time in both backward and
Forward phases of training.

Mnist can be trained with less than 2 percent loss
In less than 12 seconds on a single thread and
Less than 3 seconds using multiple threads.

This is achieved in the following ways

* Non 0 matrix values are extracted and re-ordered
* Cache optimized routines for matrix vector multiply during backprop using blocks instead of individual values
* SIMD routines for blockwise dot product and scalar vector multiplication are added
* Block sparsity is gradually added during the first few epochs which then also reduces forward multiplies