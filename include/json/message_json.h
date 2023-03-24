//
// Created by kriso on 3/12/2023.
//

#ifndef NNNN_MESSAGE_JSON_H
#define NNNN_MESSAGE_JSON_H

#include "network.h"
#include "read_mnist.h"
#include <stdlib.h>
#include <vector>
#include <fstream>
#include "nlohmann/json.hpp"

namespace noodle {
    using namespace Eigen;
    using namespace std;
    using json = nlohmann::json;

    inline num_t rounding() {
        return 1e4;
    }

    template<typename t2d>
    void write(json &j, const t2d &mat) {
        for (index_t col = 0; col < mat.cols(); ++col) {
            json jc;
            for (index_t row = 0; row < mat.rows(); ++row) {
                auto rounded = nround((double) mat(row, col), (double) rounding());
                if (abs(rounded) < 1e-5)
                    jc.push_back(0);
                else
                    jc.push_back(rounded);
            }
            j.push_back(jc);
        }
        print_dbg("saving mat/vec", mat.rows(), mat.cols());
    }

    template<typename t2d>
    void mat_load(t2d &mat, json &j) {
        index_t cols = j.size();
        index_t rows = 0;
        if (cols > 0) {
            rows = j[0].size();
        }

        print_dbg("loading", rows, cols);
        mat.resize(rows, cols);
        for (index_t col = 0; col < mat.cols(); ++col) {

            if (j[col].size() != rows) {
                fatal_err("invalid col size", j[col].size());
                return;
            }

            for (index_t row = 0; row < mat.rows(); ++row) {
                mat(row, col) = (num_t) j[col][row];
            }

        }
    }

    message load_message(json &j) {
        message r;
        r.kind = j["kind"];
        r.name = j["name"];
        if (r.kind.empty() || r.name.empty()) {
            fatal_err("name or kind not specified");
            return r;
        }
        json val = j["values"];
        if (val.empty()) {
            //fatal_err("empty values");
            return r;
        }
        print_dbg("loading values for", r.kind, r.name);
        for (auto &v: val.items()) {
            string name = v.key();
            if (v.value().is_object()) {
                print_wrn("unexpected object found");
                continue;
            }
            if (v.value().is_string()) {
                r.data[name] = (string) v.value();
            }
            if (v.value().is_number_float()) {
                r.data[name] = (num_t) v.value();
            }
            if (v.value().is_number_integer()) {
                r.data[name] = (index_t) v.value();
            }

            if (v.value().is_array()) {
                print_dbg("name", qt(name), "size", v.value().size());
                if (v.value().empty()) {
                    print_wrn("empty value");
                    continue;
                }
                if (v.value().size() == 1) {
                    vec_t dest;
                    mat_load(dest, v.value());
                    r.data[name] = dest;
                } else {
                    mat_t dest;
                    mat_load(dest, v.value());
                    r.data[name] = dest;
                }
            }
        }
        return r;
    }

    json write_message(message &m) {
        json mj;
        print_dbg("creating json message", qt(m.kind), qt(m.name));
        mj["kind"] = m.kind;
        mj["name"] = m.name;
        json val;
        for (auto &d: m.data) {
            print_dbg("saving", qt(d.first));
            if (const index_t *i = std::get_if<index_t>(&d.second)) {
                print_dbg("saving index_t", *i);
                val[d.first] = *i;
            }
            if (const string *s = std::get_if<string>(&d.second)) {
                print_dbg("saving index_t", *s);
                val[d.first] = *s;
            }
            if (const num_t *n = std::get_if<num_t>(&d.second)) {
                print_dbg("saving index_t", *n);
                val[d.first] = *n;
            }
            if (const vec_t *v = std::get_if<vec_t>(&d.second)) {
                write(val[d.first], *v);
            }
            if (const mat_t *m = std::get_if<mat_t>(&d.second)) {
                write(val[d.first], *m);
            }
        }
        mj["values"] = val;
        return mj;
    }

    template<typename TLoad>
    bool load_messages(TLoad &&load_fn, string path) {
        std::ifstream f(path);
        if (!f) {
            /// this is ok
            print_inf("data file", path, "not found");
            return false;
        }

        json content = json::parse(f);
        json data = content["data"];
        if (data.empty()) {
            print_wrn("no data found");
            return false;
        }

        if (!data.is_array()) {
            print_wrn("data should be an array");
            return false;
        }

        auto ij = data.begin();
        for (auto &ij: data) {
            if (!ij.is_object()) {
                fatal_err("invalid type");
                return false;
            }
            load_fn(load_message(ij));
        }
        return true;
    }

    bool load_messages(vector<message> &out, string path) {
        auto load_fn = [&out](const message &m) {
            out.push_back(m);
        };
        return load_messages(load_fn, path);
    }

    bool load_messages(graph &g, string path) {
        auto n = g.begin();
        auto load_fn = [&](const message &m) {
            if (n == g.end()) {
                fatal_err("incomplete data or model does not match data");

            } else {
                var_put_message(n->operation, m);
                ++n;
            }
        };
        return load_messages(load_fn, path);
    }

    /// could be done with a coroutine - ooh
    bool save_messages(vector<message> &messages, string path) {
        ofstream f(path);
        if (!f) {
            print_err("could not open file for writing", path);
            return false;
        }
        json output;
        json data_array;
        print_dbg("saving", messages.size(), "messages");
        for (auto &m: messages) {
            data_array.push_back(write_message(m));
        }
        output["data"] = data_array;
        print_inf("writing", path);
        f << output << std::endl;
        f.flush();
        return true;
    }

    void save_messages(graph &g, string path) {
        ofstream f(path);
        if (!f) {
            print_err("could not open file for writing", path);
            return;
        }
        json output;
        json data_array;
        print_dbg("saving", g.nodes.size(), "nodes");
        for (auto &n: g) {
            message m;
            var_get_message(m, n.operation);
            m.name = n.name;
            data_array.push_back(write_message(m));
        }
        output["data"] = data_array;
        print_inf("writing", path);
        f << output;
        f.flush();
    }

}
#endif //NNNN_MESSAGE_JSON_H
