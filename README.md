# Neural trainer from Scratch
I wrote a neural net for [MNIST](http://yann.lecun.com/exdb/mnist/) digit 
classification using only standard C++ headers and 
[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), a linear 
algebra library for cpp. I managed to train a neural net that achieves more than 96%
accuracy on the test set. You can try my classifier on your handwriting [here](http://oliverjiang.me/2021/08/15/neural-net.html).

## Goals
My goals were:
1. Solidify what I had learned from CS188 Artificial Intelligence.
2. Learn C++.
3. Prepare for CS189 Machine Learning, which I will be taking in Fall 2021.
- Update 2021 Oct 11: I had to write a MNIST neural net in Python for EECS189
last week. I got it to achieve 98% accuracy (2% higher than my C++ implementation), 
and it trained much faster than my C++ version.

For CS188, I had already done a project on MNIST digit classification, but a
neural net library had been provided to us which hid a lot of the details. 
Although I had gained a decent abstract understanding of neural nets, I 
quickly realized that there were many gaps in my concrete understanding. For 
example, I could hand-turn backpropagation for simple examples you might find
on an exam, but I wasn't sure about the math to vectorize backpropagation for
a neural net with hundreds of neurons. I feel that both my concrete and 
abstract understanings have both deepened after this project.

As for the second goal of learning C++, I think this project functioned as
a good sampler of the language. Coming from a background of Python and C,
learning C++ was just a matter of figuring out its quirks. Interestingly, 
I found that programming in C++ felt more similar to programming in Python
rather than in C. Despite the obvious syntactic similarities between C++ and
C, the object-oriented nature of C++ and its convenience features like strings
and vectors meant that the way I thought about my code was more similar to 
Python.

I can't really assess the success of the third goal, preparing for CS189, until
I take the class, but I am very excited to learn more about machine learning :).

## Web Demo
I made a simple web demo to show off my neural net. You can try it [here](http://oliverjiang.me/2021/08/15/neural-net.html).
The code is in the directory `web_demo`.

## Observations
- Compiler flags are very important. Compilers are something I didn't have to 
deal with when working in Python, so I didn't expect that the compiler
was the problem when I first tested my code and found it training painfully 
slowly. After profiling my code with lots of `clock()` calls and print statements,
I found that the problem was matrix multiplication, which led me to stumble
upon [this](https://stackoverflow.com/questions/36659004/eigen-matrix-multiplication-speed)
stackoverflow discussion. After adding a simple flag `-Ofast`, training
sped up almost tenfold.

- Preprocessing inputs for the web demo is extremely important. At first,
I simply lowered the resolution of the user's digit drawing to 28x28 and fed
it into the neural net, but I found that it was only slightly better than
randomly guessing; no where close to the 96% accuracy on the test set. I
remembered that all of the examples in the MNIST dataset are centered, so
I centered the inputs before feeding them into the NN, and the accuracy
shot up to around what I expected. One possible change I can make for the future
is to scale the image before feeding it to the NN as well, as currently, drawing
the digit extremely large or extremely small can trip it up.

- There are lots of bad resources on the internet. I suppose that shouldn't have
been a surprise. I recommend this [book](http://neuralnetworksanddeeplearning.com)
by Michael Nielsen and the 3Blue1Brown Youtube videos on the subject.

## Usage
To try my neural network, clone this repo and download the 
[MNIST](http://yann.lecun.com/exdb/mnist/) dataset into the `data/` directory, 
and download the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) 
library into the `include/` directory. Then, run the following commands.
```
$ sh compile.sh       # compile
$ ./a.out 300         # train for 300 epochs
```
The output will be saved in a file called `weights_and_biases.txt`. The format
is: The bias vector of the first dense_layer, followed by an empty line, followed by
the weight matrix of the first dense_layer, followed by an empty line, followed by the
bias vector of the second dense_layer, and so forth.

Depending on the endianness of your machine, you may need to comment out the
`swap_endian` lines in `read_mnist.cpp`.
