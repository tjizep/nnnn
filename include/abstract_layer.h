//
// Created by kriso on 2/9/2023.
//

#ifndef NNNN_ABSTRACT_LAYER_H
#define NNNN_ABSTRACT_LAYER_H

#include <string>
#include "message.h"

namespace noodle {
    using namespace std;

    struct abstract_layer {
        string name;

        abstract_layer(string name) : name(name) {};
    };

    /// backprop state data
    struct gradients {
        /// results for forward operation
        vec_t activation = row_vector(); /// same as input or 'x[i]'
        vector<vec_t> activations; /// same as input or 'x's
        vec_t output = row_vector(); /// as in 'y' or f('x[i]')
        vector<vec_t> errors;

        /// result for gradient backpropagation
        vec_t bp_input = row_vector(); /// aka error f'('x[i-1]')
        vec_t bp_output = row_vector(); /// f'('x[i]') or gradient

        /// gradient variables
        message variables;

    };
}
#endif //NNNN_ABSTRACT_LAYER_H
