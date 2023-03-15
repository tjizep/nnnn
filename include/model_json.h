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
#include <message_json.h>

namespace noodle{
    using namespace std;
    using json = nlohmann::json;

    bool is_spec_card(const json& s){
        return s.is_number_integer();
    }

    bool is_spec_child(const json& s){
        return s.is_array();
    }
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
        if(n.name.empty()){
            n.clear();
            fatal_err("invalid name");
        }

        return n;
    }
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
    }

    struct mnist_loader{
        void load(training_set& ts, json& def){

            index_t image_size = def["scale"];

            if(image_size <= 0){
                fatal_err("invalid image size", image_size);
            }


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

    bool json2varlayers(graph & g, VarLayers& layers, const json& l){
        if(!validate(l, {"def", "kind", "name"}))
            return false;
        if(!l["name"].is_string()){
            fatal_err("name is not a string");
            return false;
        }
        node n = from(l);
        if(l.empty()){
            fatal_err("invalid node");
            return false;
        }
        index_t nix = g.add(n); // will add a source if none exists based on order
        n = g.at(nix);


        auto l_def = l["def"];
        auto l_kind = l["kind"];

        string name = n.name;
        layer oper = noodle::empty_layer{};
        uint32_t inputs = g.find_outputs_by_source(n); /// all nodes have a input
        g.resolve(nix).inputs = inputs;


        if(l_kind == "SPARSE_FC"){
            for(auto s : n.source){
                if(g.resolve(s).empty()){
                    fatal_err("source", s, "does not exist");
                    return false;
                }
            }

            uint32_t outputs = g.find_outputs(n);
            if(inputs==0){
                fatal_err("no non-zero output source found");
                return false;
            }

            num_t sparsity = 0;
            num_t sparsity_greed = 8;
            num_t momentum = 0;
            if(l_def.contains("sparsity"))
                sparsity = l_def["sparsity"];
            oper = noodle::fc_layer{inputs, outputs, sparsity, sparsity_greed, momentum};
            g.resolve(nix).outputs = outputs;
        }

        if(l_kind == "LRELU"){
            num_t leakiness = 1000;
            if(l_def.contains("leakiness"))
                leakiness = l_def["leakiness"];
            oper = noodle::relu_layer{leakiness};
            g.resolve(nix).outputs = inputs; /// by default output size = inputsize ?
        }

        if(l_kind == "TANH"){
            oper = noodle::tanh_layer{};
            g.resolve(nix).outputs = inputs; /// by default output size = inputsize ?
        }

        if(l_kind == "SIGMOID"){
            oper = noodle::sigmoid_layer{};
            g.resolve(nix).outputs = inputs; /// by default output size = inputsize ?
        }
        if(l_kind == "SWISH"){
            num_t beta = 1;
            if(l_def.contains("beta"))
                beta = l_def["beta"];
            oper = noodle::swish_layer{beta};
            g.resolve(nix).outputs = inputs; /// by default output size = inputsize ?
        }

        if(l_kind == "FLAT_SIGMOID"){
            num_t flatness = 0.35;
            if(l_def.contains("flatness"))
                flatness = l_def["flatness"];
            oper = noodle::low_sigmoid_layer{flatness};
            g.resolve(nix).outputs = inputs; /// by default output size = inputsize ?
        }

        if(l_kind == "DROPOUT"){
            num_t rate = 0.1;
            if(l_def.contains("rate"))
                rate = l_def["rate"];
            if(rate < 0 || rate > 1){
                fatal_err("invalid dropout rate", rate);
                return false;
            }
            oper = noodle::dropout_layer{rate};
            g.resolve(nix).outputs = inputs; /// by default output size = inputsize ?
        }
        if(l_kind == "SOFTMAX"){
            oper = noodle::soft_max_layer{};
            g.resolve(nix).outputs = inputs; /// by default output size = inputsize ?
        }

        g.resolve(nix).operation = oper;
        //layers.push_back(oper);
        return true;
    }
    bool json_ensemble_2varlayers(graph& g, VarLayers& layers, const json& l){
        if(!validate(l, {"def", "kind"}))
            return false;
        auto l_def = l["def"];
        auto l_kind = l["kind"];
        layer oper = noodle::empty_layer{};
        if(l_kind == "ENSEMBLE"){
            layer_holder l;
            auto l_models = l_def["models"];
            json2varlayers(g, l.model, l_models);

            uint32_t in_size = l_def["inputs"];
            uint32_t out_size = l_def["outputs"];
            oper = noodle::ensemble<layer_holder>{l, in_size,out_size};
            //layers.push_back();
        }
        layers.push_back(oper);
        return true;
    }

    bool load_model_from_json(string path){

        std::ifstream f(path);
        if (!f) {
            fatal_err("file",path,"not found");
            return false;
        }

        json def = json::parse(f);
        if(def.empty()){
            return false;
        }
        validate(def,
             R"(
                    {"model":
                        ["name","kind",{"graph":["name", {"def":["nodes"]}]},"optimizer",
                            {"data":
                                [{"kind":1}, {"name":1}, {"outputs":1}, {"def":["scale", "path", "test_label_path"]}]
                            }
                        ]
                 })"_json);

        graph g;

        auto model_def = def["model"];

        auto optimizer_def = model_def["optimizer"];
        auto model_graph = model_def["graph"];
        auto graph_name = model_graph["name"];
        auto model_graph_def = model_graph["def"];
        auto model_nodes =  model_graph_def["nodes"];
        auto data = model_def["data"];
        auto data_kind = data["kind"];
        auto data_def = data["def"];
        auto data_dir = data_def["path"];
        string persist = model_graph_def.contains("persist") ? model_graph_def["persist"] : "";
        node dn = from(data);
        string name = dn.name;

        g.add(dn);

        training_set ts;

        if(data_kind == "mnist"){
            mnist_loader loader;
            loader.load(ts, data["def"]);
        }else{
            fatal_err("unrecognised kind of data",(string)data_kind);
            return false;
        }

        VarLayers physical;
        for(auto l : model_nodes){
            if(!json2varlayers(g, physical, l))
                return false;
            // NOT yet, json_ensemble_2varlayers(physical, l);
        }
        print_dbg("physical.size()",physical.size());

        for(auto o : optimizer_def) {
            if (!validate(o, {"kind", "def"}))
                return false;
            json kind = o["kind"];
            json def = o["def"];
            if (!validate(def, {"epochs", "mini_batch_size", "learning_rates", "threads"})) {
                return false;
            }

            print_dbg("OK optimizer");
            num_t mini_batch_size = def["mini_batch_size"];
            size_t epochs = def["epochs"];
            size_t threads = def["threads"];
            auto lr = def["learning_rates"];
            auto save_schedule = def["save_schedule"];// unused
            print_dbg("learning rates", (num_t) lr[0], (num_t) lr[1]);
            array<noodle::num_t, 2> learning_rate = {lr[0], lr[1]};
            if (!g.build_destinations()){
                return false;
            }
            noodle::trainer n(ts, mini_batch_size, learning_rate);
            if(!persist.empty()){
                print_inf("persisting to/from", qt(persist));
                load_messages(g, persist);
                auto curr_acc = n.evaluate(g,1.0);
                n.print_accuracy(curr_acc);
            }


           // n.stochastic_gradient_descent(physical, epochs, threads, 75);
            graph best = n.stochastic_gradient_descent(g, epochs, threads, 0);
            if(!persist.empty()) {
                save_messages(best, persist);
            }
            auto curr_acc = n.evaluate(best,1.0);
            n.print_accuracy(curr_acc);
        }
        return true;
    }
}
#endif //NNNN_MODEL_JSON_H
