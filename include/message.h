//
// Created by kriso on 3/6/2023.
//

#ifndef NNNN_MESSAGE_H
#define NNNN_MESSAGE_H
#include <basics.h>
namespace noodle{
    using namespace std;
    using namespace Eigen;
    typedef variant<vec_t, mat_t> v_data_t;
    struct message{
        string name;
        string kind;
        unordered_map<string, v_data_t> data;
    };
}

#endif //NNNN_MESSAGE_H
