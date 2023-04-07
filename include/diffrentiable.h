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

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;

            return output;
        }

        void bp(gradients& state, const vec_t &output_error, num_t learning_rate) {
            state.bp_output = output_error;
        }

    };
}// noodle
#endif //NNNN_DIFFRENTIABLE_H
