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
    bool contains(json& j, std::vector<string> stuff){
        for(auto s: stuff){
            if(!j.contains(s)) {
                cerr << "expected '" << s << "'" << endl;
                return false;
            }
        }
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

    void load_model_from_json(string path){

        std::ifstream f(path);
        if (!f) {
            return;
        }

        json def = json::parse(f);
        if(def.empty()){
            return;
        }

        auto model_def = def["model"];
        auto optimizer_def = model_def["optimizer"];
        auto layers = model_def["layers"];

        if(optimizer_def.empty()){
            return;
        }

        if(layers.empty()){
            return;
        }

        if(model_def.empty()){
            return;
        }

        auto data = model_def["data"];
        if(data.empty()){
            return;
        }
        if(!contains(data,{"kind", "def"}))
            return;

        auto data_kind = data["kind"];
        auto data_def = data["def"];
        auto data_dir = data_def["path"];
        if(data_kind.empty())
            return;
        if(data_dir.empty())
            return;
        if(data["def"].empty())
            return;

        training_set ts;

        if(data_kind == "mnist"){
            mnist_loader loader;
            loader.load(ts, data["def"]);
        }else{
            return;
        }

        VarLayers physical;
        for(auto l : layers){
            if(!contains(l,{"def", "kind"}))
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
                physical.push_back(noodle::fc_layer{inputs, outputs, sparsity, sparsity_greed, momentum});

            }

            if(l_kind == "LRELU"){
                num_t leakiness = 1000;
                if(l_def.contains("leakiness"))
                    leakiness = l_def["leakiness"];
                physical.push_back(noodle::relu_layer{leakiness});
            }
            if(l_kind == "SOFTMAX"){
                physical.push_back(noodle::soft_max_layer{});
            }
        }
        cout << "physical.size() " << physical.size() << endl;
        for(auto o : optimizer_def){
            if(!contains(o, {"kind","def"}) )
                return;
            json kind = o["kind"];
            json def = o["def"];
            if(!contains(def, {"epochs", "mini_batch_size", "learning_rates", "threads"}))
                return;
            cout << "OK optimizer" << endl;
            num_t mini_batch_size = def["mini_batch_size"];
            size_t epochs = def["epochs"];
            size_t threads = def["threads"];
            auto lr = def["learning_rates"];
            auto save_schedule = def["save_schedule"];// unused
            array<noodle::num_t,2>  learning_rate = {0.01, 0.0001};
            noodle::trainer n(ts, mini_batch_size, learning_rate);

            n.stochastic_gradient_descent(physical, epochs, threads, 75);


        }

    }
}
#endif //NNNN_MODEL_JSON_H
