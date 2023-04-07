//
// Created by kriso on 2/9/2023.
//

#ifndef NNNN_ABSTRACT_LAYER_H
#define NNNN_ABSTRACT_LAYER_H

#include <string>

namespace noodle {
    using namespace std;

    struct abstract_layer {
        string name;

        abstract_layer(string name) : name(name) {};
    };

    /// backprop state data
    struct gradients {
        /// results for forward operation
        vec_t activation = row_vector(); /// same as input or 'x'
        vector<vec_t> activations; /// same as input or 'x's
        vec_t output = row_vector(); /// as in 'y'
        vector<vec_t> errors;

        /// result for gradient backpropagation
        vec_t bp_input = row_vector(); // aka gradient
        vec_t bp_output = row_vector();

        /// batch gradient variables
        mat_t mini_batch_update_weights = matrix();
        vec_t mini_batch_update_biases = row_vector();


    };
}
#endif //NNNN_ABSTRACT_LAYER_H
