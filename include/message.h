//
// Created by kriso on 3/6/2023.
//

#ifndef NNNN_MESSAGE_H
#define NNNN_MESSAGE_H
#include <basics.h>
namespace noodle{
    using namespace std;
    using namespace Eigen;
    typedef variant<vec_t, mat_t, num_t, index_t, string> v_data_t;
    template<typename T>
    inline bool get_any(T& vt, const v_data_t& data){
        if (const T *n = std::get_if<T>(&data)) {

            vt = *n;
            return true;
        }
        return false;
    }
    inline bool get_vector(vec_t& v, v_data_t& vdata){
        return get_any<vec_t>(v, vdata);
    }
    inline bool get_mat(mat_t& v, v_data_t& vdata){
        return get_any<mat_t>(v, vdata);
    }
    template<typename T>
    inline T get_v(const v_data_t& vdata){
        T r;
        if(get_any<T>(r,vdata)){
            print_dbg("expected variant type in v_data_t not found");
        }
        return r;
    }

    struct message{
        string name;
        string kind;
        unordered_map<string, v_data_t> data;
    };
}

#endif //NNNN_MESSAGE_H
