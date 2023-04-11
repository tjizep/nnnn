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
    template<typename T>
    inline T& get_any_(v_data_t& data){
        if (T *n = std::get_if<T>(&data)) {

            return *n;
        }
        fatal_err("could not convert message data type");
        throw exception();
    }
    template<typename T>
    inline const T& get_any_(const v_data_t& data){
        if (const T *n = std::get_if<T>(&data)) {

            return *n;
        }
        fatal_err("could not convert message data type");
        throw exception();
    }

    inline bool get_vector(vec_t& v, v_data_t& vdata){
        return get_any<vec_t>(v, vdata);
    }
    inline vec_t& get_vector(v_data_t& vdata){
        return get_any_<vec_t>(vdata);
    }
    inline mat_t& get_mat(v_data_t& vdata){
        return get_any_<mat_t>(vdata);
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
        inline num_t& get_number(const string& name){
            if(!data.contains(name)){
                data[name] = num_t();
            }
            return get_any_<num_t>(data[name]);
        }
        inline const num_t& get_number(const string& name) const {
            auto id = data.find(name);
            if(id == data.end()){
                fatal_err("could not find",qt(name),"in message data");
            }
            return get_any_<num_t>(id->second);
        }
        inline vec_t& get_vector(const string& name){
            if(!data.contains(name)){
                data[name] = row_vector();
            }
            return get_any_<vec_t>(data[name]);
        }
        inline const vec_t& get_vector(const string& name) const {
            auto id = data.find(name);
            if(id == data.end()){
                fatal_err("could not find",qt(name),"in message data");
            }
            return get_any_<vec_t>(id->second);
        }
        inline mat_t& get_mat(const string& name){
            if(!data.contains(name)){
                data[name] = matrix();
            }
            return get_any_<mat_t>(data[name]);
        }
        inline const mat_t& get_mat(const string& name) const {
            auto id = data.find(name);
            if(id == data.end()){
                fatal_err("could not find",qt(name),"in message data");
            }
            return get_any_<mat_t>(id->second);
        }
    };
}

#endif //NNNN_MESSAGE_H
