#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_
#include <basics.h>
#include <activations.h>
#include <optimizers.h>
#include <sparsity.h>
#include <dense_layer.h>
#include <mutex>
#include <shared_mutex>
#include <thread>

namespace noodle {
    using namespace std;
    using namespace Eigen;
    typedef std::array<num_t,2> LearningRate;
    class trainer {
    public:
    private:
        typedef vector<layer> VarLayersType ;
        vector<vec_t> training_inputs_;
        vector<vec_t> training_outputs_;
        vector<vec_t> test_inputs_;
        vector<int> test_labels_;
        size_t mini_batch_size_ = 3;
        LearningRate learning_rate_ = {0.3,0.01};

    public:


        trainer(vector<vec_t> &training_inputs, vector<vec_t> &training_outputs,
                vector<vec_t> &test_inputs, vector<int> &test_labels,
                num_t mini_batch_size, std::array<num_t,2> learning_rate) {
            auto prev = 0;
            training_inputs_ = training_inputs;
            training_outputs_ = training_outputs;
            test_inputs_ = test_inputs;
            test_labels_ = test_labels;
            mini_batch_size_ = mini_batch_size;
            learning_rate_ = learning_rate;
        }

        static vec_t var_feed_forward(vec_t &a0, VarLayers& model) {
            vec_t activation = a0;
            int at = 0;
            //cout << "var input activations " << activation.norm() << "@" << (0) <<  endl;
            for (auto& l : model) {
                activation = var_forward(l,activation);
                //cout << at << " " << var_get_name (l) << " activation val " << activation.norm() << " " << activation.sum() <<endl;
                ++at;
            }
            return activation;
        }

        static inline void var_bp(const vec_t& error_, num_t lr, VarLayers& model){
            vec_t error = error_;
            int lix = model.size() - 1;
            //cout << "BACKPROP err. " << error.norm() << endl;
            for (auto cl = model.rbegin();  cl != model.rend(); ++cl) {
                error = var_layer_bp(*cl, error, lr);
                //cout << lix << " " << var_get_name (*cl) << " err val " << error.norm() << " " << error.sum() <<endl;
                //assert(!has_nan(error));
                //assert(!has_inf(error));
                --lix;
            }
        }

        VarLayers stochastic_gradient_descent(uint32_t epochs, VarLayers& model, size_t shards = 1, num_t max_streak = 3) {
            VarLayers best_model;
            array<num_t, 2> model_perf, best_perf;
            size_t best_epoch = 0;
            bool save_best = true;
            std::random_device rd;
            std::mt19937 g(rd());
            var_initialize(model);
            cout << "Beginning var stochastic gradient descent" << endl;
            auto sgd_timer = std::chrono::high_resolution_clock::now();
            vector<int> indices;
            for (uint32_t i = 0; i < training_inputs_.size(); i++) {
                indices.push_back(i);
            }
            std::uniform_int_distribution<size_t> dis(0, training_inputs_.size() - 1);
            const size_t max_batch_mirror_update = 2;
            size_t ix = 0;
            size_t total = (training_inputs_.size() * epochs)/mini_batch_size_ ;
            num_t lr_step = (learning_rate_[0] - learning_rate_[1])/epochs;
            num_t lr = learning_rate_[0];
            model_perf = evaluate(model, 0.1);
            best_perf = model_perf;
            num_t losing_streak = max_streak;

            for (uint32_t e = 0; e < epochs; e++) {
                std::shuffle(indices.begin(), indices.end(), g); // non destructive randomization
                auto epoch_timer = std::chrono::high_resolution_clock::now();
                if(shards > 1) {
                    mutex mut;
                    vector<thread> threads(shards);
                    vector<VarLayers> tmod(shards);
                    for (auto &tm: tmod)
                        tm = model;
                    for (size_t t = 0; t < shards; ++t) {
                        threads[t] = thread([&](size_t at) {
                            size_t buffer = 0;
                            for (int batch_num = 0;
                                 batch_num * mini_batch_size_ < training_inputs_.size(); batch_num++) {
                                if(batch_num % shards == 0){
                                    ++ix;
                                }
                                if (batch_num % shards == at) {

                                    update_mini_batch_single(indices, batch_num, lr, tmod[at]);

                                    unique_lock<mutex> l(mut);

                                    update_layers(model, tmod[at]);
                                    var_update_weights(model,
                                                       (num_t) ix / (num_t) total); // accumulate all the errors
                                    //tmod[at] = model;
                                    raw_copy(tmod[at], model);

                                }
                            }
                        }, t);
                    }
                    for (auto &t: threads) {
                        t.join();
                    }
                }else {
                    for (int batch_num = 0; batch_num * mini_batch_size_ < training_inputs_.size(); batch_num++) {
                        update_mini_batch(indices, batch_num, lr, model);
                        ++ix;
                    }
                }
                auto epoch_time_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> diff = epoch_time_end - epoch_timer;
                cout << "\rCompleting Epoch " << e << ". complete " <<
                     100 * ix / total
                     << "% " << " estimated acc.: " << best_perf[1] << " last epoch duration: " << diff.count() << " s" << flush;
                if(((num_t)ix / (num_t)total) > 0.06){
                    model_perf = evaluate(model, 0.15);
                    if(model_perf[1] > best_perf[1]){
                        model_perf = evaluate(model, 1);
                        if(model_perf[1] > best_perf[1]) {
                            losing_streak = max_streak;
                            best_model = model;
                            best_perf = model_perf;
                            best_epoch = e+1;
                        }else{
                            losing_streak--;
                        }
                    }else if(losing_streak <= 0) {
                        break;
                    }else{
                        losing_streak--;
                        if(lr > lr_step){
                            lr -= lr_step;
                        }
                    }
                }

            }
            cout << endl;
            model_perf = evaluate(model, 1);
            if(model_perf[1] > best_perf[1]) {
                best_model = model;
                best_perf = model_perf;
                best_epoch = epochs;
            }
            cout << "Best Epoch " << best_epoch;
            print_accuracy(",",best_perf);

            for(auto& l : best_model){
                cout << var_get_name(l) << " " << var_get_weights_norm(l) << " sparseness " << vwr_get_weights_sparseness(l) << endl;
            }

            auto epoch_time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> diff = epoch_time_end - sgd_timer;

            cout << "Finished stochastic gradient descent. Taking " << diff.count() << " seconds." << endl;
            return best_model;
        }

        num_t find_best_lr(const VarLayers &model, num_t lr_step, num_t lr) {
            //cout << endl << "best test lr " << best_test_lr << " vs. " << lr << endl;
            return lr;
        }

        /**
         *
         * @param batch_index
         * @param model
         */
         num_t loss(const vec_t& actual, const vec_t& predicted){
             vec_t r = actual - predicted;
             r = r.array().pow(2);
             return r.mean();
         }
         static inline vec_t loss_prime(const vec_t& actual, const vec_t& predicted){
             vec_t r = predicted - actual;
             r = 2.f * r / actual.size(); // standard mean squared loss
             return r;
         }
        static void update_sample(const vec_t& a0_, const vec_t& target_, size_t batch_index, num_t lr, VarLayers& model) {
            var_start_sample(model);
            vec_t a0 = a0_, target = target_;
            vec_t result = var_feed_forward(a0, model);

            vec_t error = loss_prime(target, result);

            var_bp(error, lr, model);
            var_end_sample(model);
        }
        static void update_sample(const vector<vec_t>& training_inputs_, const vector<vec_t>& training_outputs_, size_t batch_index, num_t lr, VarLayers& model) {

            vec_t a0, target;
            a0 = training_inputs_[batch_index];
            target = training_outputs_[batch_index];
            update_sample(a0, target, batch_index, lr, model);
        }

        void update_layers(VarLayers& dest, const VarLayers& source){
            auto isource = source.begin();
            auto idest = dest.begin();
            for(;isource != source.end() && idest != dest.end();++isource,++idest){
                if(!var_layer_update_bp(*idest, *isource)){
                    cerr << "layer type not found" << endl;
                }
            }
         }
        void raw_copy(VarLayers& dest, const VarLayers& source){
            auto isource = source.begin();
            auto idest = dest.begin();
            for(;isource != source.end() && idest != dest.end();++isource,++idest){
                if(!var_layer_raw_copy(*idest, *isource)){
                    cerr << "layer type not found" << endl;
                }
            }
        }

        /**
         * train a mini-batch
         * TODO: this function should be easy to multi thread without loss -having some of the weight accumulation
         * TODO: synchronized
         * @param indices array specifying order of training
         * @param batch_num number of batch in this order
         * @param lr learnin' rate
         * @param model the model that provides inference and back-prop
         */
        void update_mini_batch(vector<int> &indices, int batch_num, num_t lr, VarLayers& model) {
            int batch_index = 0;
            var_start_batch(model);
            for (int b = batch_num * mini_batch_size_;
                 b < ((batch_num * mini_batch_size_ ) + mini_batch_size_) && b < training_outputs_.size();
                 b++) {
                    batch_index = indices[b];
                    vec_t a0, target;
                    a0 = training_inputs_[batch_index];
                    target = training_outputs_[batch_index];
                    update_sample(a0, target, batch_index, lr, model);

            }
            var_end_batch(model);
            var_update_weights(model, mini_batch_size_); // accumulate all the errors

        }
        void update_mini_batch_single(vector<int> &indices, int batch_num, num_t lr, VarLayers& model) {
            int batch_index = 0;
            vec_t a0, target;
            var_start_batch(model);
            for (int b = batch_num * mini_batch_size_;
                 b < ((batch_num * mini_batch_size_ ) + mini_batch_size_) && b < training_outputs_.size();
                 b++) {
                    batch_index = indices[b];
                    vec_t a0, target;
                    a0 = training_inputs_[batch_index];
                    target = training_outputs_[batch_index];
                    update_sample(a0, target, batch_index, lr, model);
            }
            var_end_batch(model);
        }

        void print_training_data(){
            cout << "Training set in_size " << training_inputs_.size() << " ";
            cout << "Testing set in_size " << test_labels_.size() << endl;
        }

        void print_accuracy(std::string pref, array<num_t,2> result){

            cout << pref <<  " Accuracy: Train = " << result[0]*100 << " %, ";
            cout << "Validation = " << result[1]*100 << " %" << endl;
        }
        /**
         * evaluate model stochastically
         * @param model the model to evaluate
         * @param fraction of test and training inputs to evaluate, fraction can be larger than 1 if you like
         * @return { test accuracy, training accuracy }
         */
        // stochastic evaluation
        array<num_t,2> evaluate(VarLayers& model, num_t fraction_ = 1) {
            num_t fraction = abs(fraction_);
            if(fraction > 2) fraction = 1;
            std::random_device rd;
            std::mt19937 g(rd());

            array<num_t,2> result;
            size_t num_correct = 0;
            size_t output;
            size_t train_output;
            var_set_training(model, false);
            std::uniform_int_distribution<size_t> dis_t(0, training_outputs_.size() - 1);
            for (uint32_t i = 0; i < training_outputs_.size()*fraction; i++) {
                size_t o_index = dis_t(g);
                var_feed_forward(training_inputs_[o_index],model);
                vec_t vi = var_get_input(model.back());
                vi.maxCoeff(&output);
                training_outputs_[o_index].maxCoeff(&train_output);
                if (output == train_output) {
                    num_correct++;
                }
            }

            std::uniform_int_distribution<size_t> dis(0, test_inputs_.size() - 1);

            result[0] = (num_t)num_correct / (fraction*training_inputs_.size());
            num_correct = 0;
            for (uint32_t i = 0; i < test_labels_.size()*fraction; i++) {
                size_t sample_index = dis(g);
                var_feed_forward(test_inputs_[sample_index], model);
                var_get_input(model.back()).maxCoeff(&output);
                if (output == test_labels_[sample_index]) {
                    num_correct++;
                }
            }

            result[1] = (num_t)num_correct / (fraction*test_labels_.size());

            var_set_training(model, true);
            return result;
        }

        int save_weights_and_biases(string filepath) {
            std::ofstream file(filepath);
            if (file.is_open()) {
#if 0
                for(auto& l : layers){
                    file << l.biases << endl << endl;
                    file << l.weights << endl << endl;
                }
#endif
                cout << "Successfully saved to " << filepath << "." << endl;
                return 0;
            }
            cout << "Failed to save to " << filepath << "." << endl;
            return 1;
        }

    };

}

#endif