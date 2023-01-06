//
// Created by Pretorius, Christiaan on 2022-10-01.
//

#ifndef NNNN_ACTIVATIONS_H
#define NNNN_ACTIVATIONS_H
#include <basics.h>
namespace noodle {
    using namespace std;
    using namespace Eigen;


    struct sigmoid {
        num_t operator()(num_t x) const {
            return 1 / (1 + exp( -x));
        }
    };

    struct sigmoid_derivative {
        num_t operator()(num_t x) const {
            return  x * (1 - x);
        }
    };

    struct low_sigmoid {
        num_t flatness = 0.77;
        low_sigmoid(num_t flatness) : flatness(flatness){};
        num_t operator()(num_t x) const {
            return 1 / (1 + exp(-flatness * x));
        }
    };

    struct low_sigmoid_derivative {
        num_t flatness = 0.77;
        low_sigmoid_derivative(num_t flatness) : flatness(flatness){};
        num_t operator()(num_t x) const {
            return (1/flatness) * x * (1 - x);
        }
    };

    struct tanh_activation {
        num_t operator()(num_t x) const {
            return (2 / (1 + exp(-2 * x))) - 1;
        }
    };

    struct tanh_derivative {
        num_t operator()(num_t x) const {
            return 1.0f - (x*x);
        }
    };

    struct relum {
        num_t max_val = 1000;
        num_t operator()(num_t x) const {
            return (x >= 0) ? x : x/max_val;
        }
    };

    struct relu_derivative {
        num_t max_val = 1000;
        num_t operator()(num_t x) const {
            return (x >= 0) ? 1 : x/1000;
        }
    };


}
#endif //NNNN_ACTIVATIONS_H
