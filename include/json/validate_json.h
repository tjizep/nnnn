//
// Created by kriso on 3/23/2023.
//

#ifndef NNNN_VALIDATE_JSON_H
#define NNNN_VALIDATE_JSON_H
#include <stdlib.h>
#include <vector>
#include <fstream>
#include "nlohmann/json.hpp"
#include "basics.h"

namespace noodle {
    using namespace std;
    using json = nlohmann::json;
    bool validate_name(string name, size_t count,const json& j){
        if (!j.contains(name)) {
            fatal_err("expected element", qt(name), "not found");
            return false;
        } else if (j[name].size() < count) {
            fatal_err("element", qt(name), "has less than", count, "elements");
            return false;
        }
        return true;
    }

    bool validate(const json & j, const json& spec, size_t lvl = 0){

        for(auto e: spec.items()){
            size_t count = 0;
            if(spec.is_object()) {

                if (e.value().is_number_integer()) {
                    count = e.value();
                }
                if(j.is_array()){

                }else{
                    if (!validate_name(e.key(), count, j)) {
                        return false;
                    }
                }

                validate(j[e.key()], e.value(), lvl + 1);
            }else if(spec.is_array()){
                if(e.value().is_string()){
                    if(j.is_array()){

                    }else{
                        if (!validate_name(e.value(), count, j)) {
                            return false;
                        }
                    }

                }else if(e.value().is_object()){
                    if(j.is_array()){

                    }else
                        validate(j, e.value(), lvl + 1);
                }
            }
        }
        return true;
    }

}
#endif //NNNN_VALIDATE_JSON_H
