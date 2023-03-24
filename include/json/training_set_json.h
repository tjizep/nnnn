//
// Created by kriso on 3/18/2023.
//

#ifndef NNNN_TRAINING_SET_JSON_H
#define NNNN_TRAINING_SET_JSON_H
#include "message_json.h"
#include "validate_json.h"

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

    static bool load_training_set(training_set& ts, string path, string format){
        vector<message> messages;
        if(!load_messages(messages, path, format)){
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
    static void test_training_set(const training_set& ts, string path, string format = "json");

    static bool save_training_set(const json& def, const training_set& ts){
        if(ts.training_outputs.size() != ts.training_outputs.size()){
            fatal_err("training inputs are not the same as the outputs");
            return false;
        }
        validate(def, {"path","format"});
        string path = def["path"];
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
        auto format = def["format"];
        if(format.is_array()){
            for(auto &f : format.items()){
                if(f.value().is_string()){
                    string sformat = f.value();
                    if(!save_messages(messages, path, sformat)){
                        return false;
                    }
                    test_training_set(ts, path, sformat);
                }else{
                    fatal_err("invalid training set export format");
                    return false;
                }
            }
        }else if (format.is_string()){
            string sformat = format;
            if(!save_messages(messages, path, sformat)){
                return false;
            }
            test_training_set(ts, path, sformat);
        }else{
            fatal_err("invalid training set export format");
            return false;
        }

        return true;
    }
    static void test_training_set(const training_set& ts, string path, string format){
        print_inf("training set test",qt(path),qt(format));
        training_set ts_test;
        load_training_set(ts_test, path, format);
        if(ts_test.training_inputs.size() != ts.training_inputs.size()){
            fatal_err("training set training_inputs",ts_test.training_inputs.size(), ts.training_inputs.size());
        }
        if(ts_test.training_outputs.size() != ts.training_outputs.size()){
            fatal_err("training set training_outputs",ts_test.training_outputs.size(), ts.training_outputs.size());
        }
        if(ts_test.training_labels.size() != ts.training_labels.size()){
            fatal_err("training set training_labels",ts_test.training_labels.size(), ts.training_labels.size());
        }
        if(ts_test.test_inputs.size() != ts.test_inputs.size()){
            fatal_err("training set test_inputs",ts_test.test_inputs.size(), ts.test_inputs.size());
        }
        if(ts_test.test_outputs.size() != ts.test_outputs.size()){
            fatal_err("training set test_outputs",ts_test.test_outputs.size(), ts.test_outputs.size());
        }
        if(ts_test.test_labels.size() != ts.test_labels.size()){
            fatal_err("training set test_labels",ts_test.test_labels.size(), ts.test_labels.size());
        }
    }
}
#endif //NNNN_TRAINING_SET_JSON_H
