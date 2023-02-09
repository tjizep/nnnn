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
}
#endif //NNNN_ABSTRACT_LAYER_H
