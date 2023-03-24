//
// Created by kriso on 3/23/2023.
//

#ifndef NNNN_INFERENCE_JSON_H
#define NNNN_INFERENCE_JSON_H
#include <stdlib.h>
#include <vector>
#include <fstream>
#include "nlohmann/json.hpp"
#include "basics.h"
#include "model.h"
#include "graph.h"
#include "message_json.h"
#include "validate_json.h"
#include "node_json.h"

namespace noodle{
    using namespace Eigen;
    using namespace std;
    using json = nlohmann::json;
    bool infer_from_json(json& model_def, json& model_def_name, graph& g){
        if(model_def.contains("inference")) {
            validate(model_def, R"({"inference":["kind","name",{"def":["input","output"]}]})"_json);
            auto inference = model_def["inference"];
            auto inference_def = inference["def"];
            node inf_n = from(inference);
            if(inf_n.empty()){
                fatal_err("node","'inference'","is invalid");
                return false;
            }
            if (!inf_n.enabled) {
                return true; // not an error
            }
            string input;
            string output;
            if (inference_def["input"].is_string()) {
                input = inference_def["input"];
            } else {
                fatal_err("inference input should be a string");
                return false;
            }
            if (inference_def["output"].is_string()) {
                output = inference_def["output"];
            } else {
                fatal_err("inference output should be a string");
                return false;
            }
            ofstream foutput(output);
            if (!foutput) {
                fatal_err("could not open", qt(output), "for writing");
                return false;
            }
            auto load_fn = [&model_def_name](const message &m) {
                string name = model_def_name;
                if (m.kind == "INFER" && !name.empty()) {
                    print_inf("performing inference on", qt(m.name));
                    if (m.data.contains(name)) {
                        print_inf("ok to do inference for", qt(name));
                    } else {
                        print_wrn("could not find inference vector or matrix");
                    }
                }
            };
            load_messages(load_fn, input);
        }

        return true;
    }
}
#endif //NNNN_INFERENCE_JSON_H
