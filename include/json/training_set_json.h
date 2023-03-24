//
// Created by kriso on 3/18/2023.
//

#ifndef NNNN_TRAINING_SET_JSON_H
#define NNNN_TRAINING_SET_JSON_H
#include "message_json.h"
namespace noodle{
    using namespace Eigen;
    using namespace std;
    using json = nlohmann::json;
    template<typename MapT, typename ValueT>
    inline bool contains(const MapT& m, const vector<ValueT>& keys){
        for(auto& k: keys){
            if(!m.contains(k)){
                return false;
            }
        }
        return true;
    }

    bool load_training_set(training_set& ts, string path){
        vector<message> messages;
        if(!load_messages(messages, path)){
            return false;
        }
        if(messages.empty()){
            print_wrn("no data found in",path);
            return false;
        }
        message &meta = *(messages.begin());
        if(meta.kind != "TRAINING_SET"){
            print_wrn("found kind",qt(meta.kind),", instead of 'TRAIN' in",path);
            return false;
        }
        if(!contains<typeof(meta.data), string>(meta.data,{"TRAIN_OUT","TRAIN_IN","TEST_IN","TEST_LABELS"})){
            print_wrn("could not find one of","TRAIN_OUT","TRAIN_IN","TEST_IN","TEST_LABELS");
            return false;
        }
        auto mi = messages.begin();
        ++mi;
        if(mi == messages.end()){
            print_wrn("data in",path,"is incomplete (only meta data found)");
            return false;
        }
        for(;mi != messages.end();++mi){
            if(mi->kind == "TRAIN"){

                vec_t temp = row_vector();
                if(mi->data.contains("IN")) {
                    auto vi = mi->data["IN"];
                    if (!get_any(temp, vi)) {
                        print_wrn("expected vector data for train input");
                        return false;
                    }
                }
                if(mi->data.contains("OUT")){
                    auto vo = mi->data["OUT"];
                    ts.training_inputs.push_back(temp);
                    if(!get_any(temp, vo)){
                        print_wrn("expected vector data for train output/target");
                        return false;
                    }
                    ts.training_outputs.push_back(temp);
                }

                if(mi->data.contains("LABEL")) {
                    auto vl = mi->data["LABEL"];
                    index_t ti;
                    if (!get_any(ti, vl)) {
                        print_wrn("expected vector data for train training_labels");
                        return false;
                    }
                    ts.training_labels.push_back(ti);
                }

            }else if(mi->kind == "TEST"){


                vec_t temp = row_vector();
                if(mi->data.contains("OUT")){
                    auto vo = mi->data["OUT"];
                    if(!get_any(temp, vo)){
                        print_wrn("expected vector data for test outputs");
                        return false;
                    }
                    ts.test_outputs.push_back(temp);
                }
                if(mi->data.contains("IN")){
                    auto vi = mi->data["IN"];
                    if(!get_any(temp, vi)){
                        print_wrn("expected vector data for test outputs");
                        return false;
                    }
                    ts.test_inputs.push_back(temp);
                }
                if(mi->data.contains("LABEL")){
                    auto vl = mi->data["LABEL"];
                    index_t ti;
                    if (!get_any(ti, vl)) {
                        print_wrn("expected integer data for labels");
                        return false;
                    }
                    ts.test_labels.push_back(ti);
                }
            }else {
                print_wrn("found unexpected kind of message",mi->kind);
            }
        }
        /// {"TRAIN_OUT","TRAIN_IN","TRAIN_LABELS","TEST_IN","TEST_LABELS"}

        if(ts.test_labels.size() != get_v<index_t>(meta.data["TEST_LABELS"])){
            print_wrn("test labels are inconsistent with meta data found",ts.test_labels.size(),"expected", get_v<index_t>(meta.data["TEST_LABELS"]));
            return false;
        }
        if(ts.test_labels.size() != get_v<index_t>(meta.data["TEST_IN"])){
            print_wrn("test inputs are inconsistent with meta data found",ts.test_inputs.size(),"expected", get_v<index_t>(meta.data["TEST_IN"]));
            return false;
        }
        if(ts.test_labels.size() != get_v<index_t>(meta.data["TEST_OUT"])){
            print_wrn("test inputs are inconsistent with meta data found",ts.test_outputs.size(),"expected", get_v<index_t>(meta.data["TEST_IN"]));
            return false;
        }

        if(ts.training_inputs.size() != get_v<index_t>(meta.data["TRAIN_IN"])){
            print_wrn("train data is inconsistent with meta data found",ts.training_inputs.size(),"expected", get_v<index_t>(meta.data["TRAIN_IN"]));
            return false;
        }
        if(ts.training_labels.size() != get_v<index_t>(meta.data["TRAIN_LABELS"])){
            print_wrn("train data is inconsistent with meta data found",ts.training_labels.size(),"expected", get_v<index_t>(meta.data["TRAIN_IN"]));
            return false;
        }

        if(ts.training_outputs.size() != get_v<index_t>(meta.data["TRAIN_OUT"])){
            print_wrn("train inputs are inconsistent with meta data found",ts.training_outputs.size(),"expected", get_v<index_t>(meta.data["TRAIN_OUT"]));
            return false;
        }

        return true;
    }
    struct save_ts_options{
        index_t columns = 0;
        index_t rows = 1;
    };


    bool save_training_set(string path, training_set& ts, save_ts_options options = save_ts_options()){
        if(ts.training_outputs.size() != ts.training_outputs.size()){
            fatal_err("training inputs are not the same as the outputs");
            return false;
        }
        print_inf("saving train data to",path);
        vector<message> messages;
        index_t counter = 0;
        message meta;
        meta.name = path;
        meta.kind = "TRAINING_SET";
        meta.data["TRAIN_OUT"] = (index_t)ts.training_outputs.size();
        meta.data["TRAIN_IN"] = (index_t)ts.training_inputs.size();
        meta.data["TRAIN_LABELS"] = (index_t)ts.training_labels.size();

        meta.data["TEST_IN"] = (index_t)ts.test_inputs.size();
        meta.data["TEST_OUT"] = (index_t)ts.test_outputs.size();
        meta.data["TEST_LABELS"] = (index_t)ts.test_labels.size();
        messages.push_back(meta);
        for(auto& to : ts.training_inputs){
            message training;
            training.name = "TR_" + to_string(counter);
            training.kind = "TRAIN";
            training.data["IN"] = to;
            if(counter < ts.training_outputs.size())
                training.data["OUT"] = ts.training_outputs[counter];
            if(counter < ts.training_labels.size())
                training.data["LABEL"] = ts.training_labels[counter];

            ++counter;
            messages.push_back(training);
        }
        counter = 0;
        for(auto& ti : ts.test_inputs){
            message test;
            test.name = "TE_" + to_string(counter);
            test.kind = "TEST";
            test.data["IN"] = ti;
            if(counter < ts.test_outputs.size())
                test.data["OUT"] = ts.test_outputs[counter];
            if(counter < ts.test_labels.size())
                test.data["LABEL"] = ts.test_labels[counter];
            ++counter;
            messages.push_back(test);
        }

        return save_messages(messages, path);
    }
}
#endif //NNNN_TRAINING_SET_JSON_H
