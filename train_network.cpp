
#include <network.h>
#include <read_mnist.h>
#include <stdlib.h>
#include <model_json.h>

int main(int argc, char *argv[]) {
    using namespace std;

    noodle::load_model_from_json("../models/mnist_fc.json");
#ifdef EIGEN_VECTORIZE
    cout << "'Eigen' Vectorization is enabled" << endl;
#else
    cout << "'Eigen' Vectorization is disabled" << endl;
#endif

    return 0;
}
