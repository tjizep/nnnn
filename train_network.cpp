
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
    if(argc > 1 && argv[1]){
        print_inf("model file",argv[1]);
        noodle::load_model_from_json(argv[1]);
    }

    return -1;
}
