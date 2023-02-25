//
// Created by kriso on 2/19/2023.
//

#ifndef NNNN_DIFFRENTIABLE_H
#define NNNN_DIFFRENTIABLE_H
#include <basics.h>
#include <abstract_layer.h>
namespace noodle {
    using namespace std;
    using namespace Eigen;
    struct multiply_layer : public abstract_layer {

        vec_t input = row_vector();
        vec_t output = row_vector();

        multiply_layer() : abstract_layer("MULTIPLY") {
        }

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;

            return output;
        }

        vec_t bp(const vec_t &output_error, num_t learning_rate) {

            vec_t result = output_error;

            return result;
        }

    };
}// noodle
#endif //NNNN_DIFFRENTIABLE_H
