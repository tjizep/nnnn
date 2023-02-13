
#include <network.h>
#include <read_mnist.h>
#include <stdlib.h>
#include <model_json.h>

int main(int argc, char *argv[]) {
    using namespace std;
#ifdef EIGEN_VECTORIZE
    print_inf("'Eigen' Vectorization is enabled");
#else
    print_inf("'Eigen' Vectorization is disabled");
#endif
    noodle::load_model_from_json("../models/mnist_fc.json");


    return 0;
}
