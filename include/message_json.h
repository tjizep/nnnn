//
// Created by kriso on 3/12/2023.
//

#ifndef NNNN_MESSAGE_JSON_H
#define NNNN_MESSAGE_JSON_H
#include <network.h>
#include <read_mnist.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

namespace noodle {
    using namespace Eigen;
    using namespace std;
    using json = nlohmann::json;
    message load_message(json& j){
        message r;

        return r;
    }
    template<typename t2d>
    void load(t2d& mat, json& j){
        index_t rows = j.size();
        index_t cols = 0;
        if(rows > 0){
            cols = j[0].size();
        }
        mat.resize(rows,cols);
        for(index_t row = 0;row < mat.rows();++row){
            for(index_t col = 0;col < mat.cols();++col){
                if(j[row].size() != cols){
                    fatal_err("invalid row size",j[row].size());
                    return;
                }
                mat(row,col) = (num_t)j[row][col];
            }
        }
    }
    template<typename t2d>
    void write(json& j, const t2d& mat){
        for(index_t row = 0;row < mat.rows();++row){
            json jr;
            for(index_t col = 0;col < mat.cols();++col){
                jr.push_back((num_t)(mat(row,col)));
            }
            j.push_back(jr);
        }
    }
    json write_message(message& m){
        json mj;
        print_dbg("creating json message",m.kind,m.name);
        mj["kind"] = m.kind;
        mj["name"] = m.name;
        json val;
        for(auto& d: m.data){
            if(const index_t*  i= std::get_if<index_t>(&d.second)){
                val[d.first] = *i;
            }
            if(const string*  s= std::get_if<string>(&d.second)){
                val[d.first] = *s;
            }
            if(const num_t*  n= std::get_if<num_t>(&d.second)){
                val[d.first] = *n;
            }
            if(const vec_t*  v= std::get_if<vec_t>(&d.second)){
                write(val[d.first],*v);
            }
            if(const mat_t*  m= std::get_if<mat_t>(&d.second)){
                write(val[d.first],*m);
            }
        }
        mj["values"] = val;
        return mj;
    }
    message read_message(json& mj){
        message m;
        m.kind =  mj["kind"];
        m.name = mj["name"];
        json val = mj["values"];
        for(auto& v: val){
            if(v.is_object()){
            }
        }
        mj["values"] = val;
        return m;
    }
    void load_messages(string path, json& j, graph& g){
        std::ifstream f(path);
        if (!f) {
            /// this is ok
            print_inf("data file",path,"not found");
            return;
        }

        json content = json::parse(f);
        json data = content["data"];
        if(data.empty()){
            print_wrn("no data found");
            return;
        }

        if(!data.is_array()){
            print_wrn("data should be an array");
            return;
        }

        auto ij = data.begin();
        for(auto & n: g){
            if(ij == data.end()){
                fatal_err("incomplete data or model does not match data");
                return;
            }
            if(!ij->is_object()){
                fatal_err("invalid type");
                return;
            }

            var_put_message(n.operation, load_message(*ij));
            ++ij;
        }
    }

    void save_messages(graph& g, string path){
        ofstream f(path);
        if(!f){
            print_err("could not open file for writing",path);
            return;
        }
        json output;
        json data_array;
        for(auto & n: g){
            message m;
            var_get_message(m, n.operation);
            m.name = n.name;
            data_array.push_back(write_message(m));
        }
        output["data"] = data_array;
        print_inf("writing",path);
        f << output;
    }

}
#endif //NNNN_MESSAGE_JSON_H
