//
// Created by kriso on 2/11/2023.
//

#ifndef NNNN_MODEL_JSON_H
#define NNNN_MODEL_JSON_H

#include <network.h>
#include <read_mnist.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

namespace noodle{
    using namespace std;
    using json = nlohmann::json;
    bool is_spec_card(const json& s){
        return s.is_number_integer();
    }

    bool is_spec_child(const json& s){
        return s.is_array();
    }
    bool validate_name(string name, size_t count,const json& j){
        if (!j.contains(name)) {
            print_err("expected element", name, "not found");
            return false;
        } else if (j[name].size() < count) {
            print_err("element", name, "has less than", count, "elements");
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
                if (!validate_name(e.key(), count, j)) {
                    return false;
                }
                validate(j[e.key()], e.value(), lvl + 1);
            }else if(spec.is_array()){
                if(e.value().is_string()){
                    if (!validate_name(e.value(), count, j)) {
                        return false;
                    }
                }else if(e.value().is_object()){
                    validate(j, e.value(), lvl + 1);
                }
            }
        }
        return true;
#if 0


#endif
        return true;
    }

    struct mnist_loader{
        void load(training_set& ts, json& def){

            int32_t image_size = def["scale"];

            if(image_size <= 0)
                return;

            string data_dir = def["path"];
            string test_label_path = def["test_label_path"];
            string train_label_path = def["train_label_path"];
            string test_data_path = def["test_data_path"];
            string train_data_path = def["train_data_path"];

            ts.training_labels = mnist::get_labels(data_dir+"train-labels-idx1-ubyte");
            ts.training_inputs = mnist::get_images<noodle::vec_t>(data_dir+"train-images-idx3-ubyte",image_size);
            ts.training_outputs = mnist::get_output_vectors<noodle::vec_t>(ts.training_labels);

            ts.test_labels = mnist::get_labels(data_dir+"t10k-labels-idx1-ubyte");
            ts.test_inputs = mnist::get_images<noodle::vec_t>(data_dir+"t10k-images-idx3-ubyte",image_size);
            ts.data_size = image_size;

        }
    };

    void json2varlayers(VarLayers& layers, const json& l){
        if(!validate(l, {"def", "kind"}))
            return;

        auto l_def = l["def"];
        auto l_kind = l["kind"];
        if(l_kind == "SPARSE_FC"){
            if(l_def["inputs"].empty())
                return;
            if(l_def["outputs"].empty())
                return;
            uint32_t inputs = l_def["inputs"];
            uint32_t outputs = l_def["outputs"];

            num_t sparsity = 0;
            num_t sparsity_greed = 8;
            num_t momentum = 0;
            if(l_def.contains("sparsity"))
                sparsity = l_def["sparsity"];
            layers.push_back(noodle::fc_layer{inputs, outputs, sparsity, sparsity_greed, momentum});

        }

        if(l_kind == "LRELU"){
            num_t leakiness = 1000;
            if(l_def.contains("leakiness"))
                leakiness = l_def["leakiness"];
            layers.push_back(noodle::relu_layer{leakiness});
        }
        if(l_kind == "DROPOUT"){

            num_t rate = 0.1;
            if(l_def.contains("rate"))
                rate = l_def["rate"];
            if(rate < 0 || rate > 1){
                print_err("invalid dropout rate", rate);
                return;
            }
            layers.push_back(noodle::dropout_layer{rate});
        }
        if(l_kind == "SOFTMAX"){
            layers.push_back(noodle::soft_max_layer{});
        }
    }

    void json_ensemble_2varlayers(VarLayers& layers, const json& l){
        if(!validate(l, {"def", "kind"}))
            return;

        auto l_def = l["def"];
        auto l_kind = l["kind"];
        if(l_kind == "ENSEMBLE"){
            layer_holder l;
            auto l_models = l_def["models"];
            json2varlayers(l.model, l_models);

            uint32_t in_size = l_def["inputs"];
            uint32_t out_size = l_def["outputs"];

            layers.push_back(noodle::ensemble<layer_holder>{l, in_size,out_size});
        }
    }

    void load_model_from_json(string path){

        std::ifstream f(path);
        if (!f) {
            return;
        }

        json def = json::parse(f);
        if(def.empty()){
            return;
        }
        validate(def,
             R"(
                    {"model":
                        ["name","type","layers","optimizer",
                            {"data":
                                [{"kind":1}, {"def":["scale", "path", "test_label_path"]}]
                            }
                        ]
                 })"_json);

        auto model_def = def["model"];
        auto optimizer_def = model_def["optimizer"];
        auto layers = model_def["layers"];
        auto data = model_def["data"];
        auto data_kind = data["kind"];
        auto data_def = data["def"];
        auto data_dir = data_def["path"];

        training_set ts;

        if(data_kind == "mnist"){
            mnist_loader loader;
            loader.load(ts, data["def"]);
        }else{
            return;
        }

        VarLayers physical;
        for(auto l : layers){
            json2varlayers(physical, l);
            json_ensemble_2varlayers(physical, l);
        }
        print_dbg("physical.size()",physical.size());
        for(auto o : optimizer_def){
            if(!validate(o, {"kind", "def"}) )
                return;
            json kind = o["kind"];
            json def = o["def"];
            if(!validate(def, {"epochs", "mini_batch_size", "learning_rates", "threads"})){
                return;
            }

            print_dbg("OK optimizer");
            num_t mini_batch_size = def["mini_batch_size"];
            size_t epochs = def["epochs"];
            size_t threads = def["threads"];
            auto lr = def["learning_rates"];
            auto save_schedule = def["save_schedule"];// unused
            print_dbg("learning rates",(num_t)lr[0],(num_t)lr[1]);
            array<noodle::num_t,2>  learning_rate = {lr[0], lr[1]};
            noodle::trainer n(ts, mini_batch_size, learning_rate);

            n.stochastic_gradient_descent(physical, epochs, threads, 75);


        }

    }
}
#endif //NNNN_MODEL_JSON_H
