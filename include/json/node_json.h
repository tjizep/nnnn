//
// Created by kriso on 3/23/2023.
//

#ifndef NNNN_NODE_JSON_H
#define NNNN_NODE_JSON_H
#include <vector>
#include <fstream>
#include "nlohmann/json.hpp"
#include "basics.h"
#include "graph.h"

namespace noodle{
    using namespace Eigen;
    using namespace std;
    using json = nlohmann::json;
    node from(const json& j){
        node n;
        if(j.contains("outputs") && j["outputs"].is_number_integer())
            n.outputs = j["outputs"];
        if(j.contains("name") && j["name"].is_string())
            n.name = j["name"];
        if(j.contains("source") && j["source"].is_string())
            n.source.push_back(j["source"]);
        if(j.contains("source") && j["source"].is_array()){
            for(auto s : j["source"]){
                n.source.push_back(s);
            }
        }
        if(j.contains("enabled") && j["enabled"].is_boolean())
            n.enabled = j["enabled"];

        if(n.name.empty()){
            n.clear();
            fatal_err("invalid name");
        }

        return n;
    }
}
#endif //NNNN_NODE_JSON_H
