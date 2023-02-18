#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include <basics.h>
#include <activations.h>
#include <optimizers.h>

#include <dense_layer.h>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <model.h>
#include <ensemble.h>

namespace noodle {
    using namespace std;
    using namespace Eigen;
    struct training_set {
        vector<int> training_labels;
        vector<noodle::vec_t> training_inputs ;
        vector<noodle::vec_t> training_outputs ;
        vector<int> test_labels ;
        vector<noodle::vec_t> test_inputs;
        int32_t data_size;
    };

    typedef std::array<num_t, 2> LearningRate;

    class trainer {
    public:
    private:
        typedef vector<layer> VarLayersType;
        training_set data;
        size_t mini_batch_size_ = 3;
        LearningRate learning_rate_ = {0.3, 0.01};

    public:


        trainer(training_set &data, num_t mini_batch_size, std::array<num_t, 2> learning_rate) {
            auto prev = 0;
            this->data = data;
            mini_batch_size_ = mini_batch_size;
            learning_rate_ = learning_rate;
        }

        static vec_t var_feed_forward(vec_t &a0, VarLayers &model) {
            vec_t activation = a0;
            int at = 0;
            //cout << "var input activations " << activation.norm() << "@" << (0) <<  endl;
            for (auto &l: model) {
                activation = var_forward(l, activation);
                //cout << at << " " << var_get_name (l) << " activation val " << activation.norm() << " " << activation.sum() <<endl;
                ++at;
            }
            return activation;
        }

        static inline void var_bp(const vec_t &error_, num_t lr, VarLayers &model) {
            vec_t error = error_;
            int lix = model.size() - 1;
            //cout << "BACKPROP err. " << error.norm() << endl;
            for (auto cl = model.rbegin(); cl != model.rend(); ++cl) {
                error = var_layer_bp(*cl, error, lr);
                //cout << lix << " " << var_get_name (*cl) << " err val " << error.norm() << " " << error.sum() <<endl;
                //assert(!has_nan(error));
                //assert(!has_inf(error));
                --lix;
            }
        }

        VarLayers
        stochastic_gradient_descent(VarLayers &model, uint32_t epochs, size_t shards = 1, num_t max_streak = 3) {
            VarLayers best_model;
            array<num_t, 2> model_perf, best_perf;
            size_t best_epoch = 0;
            bool save_best = true;
            std::random_device rd;
            std::mt19937 g(rd());
            var_initialize(model);
            print_inf("Beginning var stochastic gradient descent");
            auto sgd_timer = std::chrono::high_resolution_clock::now();
            vector<int> indices;
            for (uint32_t i = 0; i < data.training_inputs.size(); i++) {
                indices.push_back(i);
            }
            std::uniform_int_distribution<size_t> dis(0, data.training_inputs.size() - 1);
            const size_t max_batch_mirror_update = 2;
            size_t ix = 0;
            const num_t lr_min = std::min<num_t>(learning_rate_[0], learning_rate_[1]);
            const num_t lr_max = std::max<num_t>(learning_rate_[0], learning_rate_[1]);
            size_t total = (data.training_inputs.size() * epochs) / mini_batch_size_;
            num_t lr_step = (lr_max - lr_min) / epochs;
            num_t lr = learning_rate_[0];

            model_perf = evaluate(model, 0.1);
            best_perf = model_perf;
            num_t losing_streak = max_streak;

            for (uint32_t e = 0; e < epochs; e++) {
                std::shuffle(indices.begin(), indices.end(), g); // non destructive randomization
                auto epoch_timer = std::chrono::high_resolution_clock::now();
                if (shards > 1) {
                    mutex mut;
                    vector <thread > threads(shards);
                    vector <VarLayers > tmod(shards);
                    for (auto &tm: tmod)
                        tm = model;
                    for (size_t t = 0; t < shards; ++t) {
                        threads[t] = thread([&](size_t at) {
                            size_t buffer = 0;
                            for (int batch_num = 0;
                                 batch_num * mini_batch_size_ < data.training_inputs.size(); batch_num++) {
                                if (batch_num % shards == 0) {
                                    ++ix;
                                }
                                if (batch_num % shards == at) {

                                    update_mini_batch_single(indices, batch_num, lr, tmod[at]);

                                    unique_lock<mutex> l(mut);

                                    update_layers(model, tmod[at]);
                                    var_update_weights(model,
                                                       (num_t) ix / (num_t) total); // accumulate all the errors
                                    tmod[at] = model;
                                    //raw_copy(tmod[at], model);

                                }
                            }
                        }, t);
                    }
                    for (auto &t: threads) {
                        t.join();
                    }
                } else {
                    for (int batch_num = 0; batch_num * mini_batch_size_ < data.training_inputs.size(); batch_num++) {
                        update_mini_batch(indices, batch_num, lr, model);
                        ++ix;
                    }
                }
                auto epoch_time_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> diff = epoch_time_end - epoch_timer;
                print_inf("Completing Epoch",e,". complete ",
                     100 * ix / total, "%");
                print_inf("estimated acc.: ",best_perf[1]);
                print_inf("last epoch duration:",diff.count(),"s, lr:",lr);
                if (((num_t) ix / (num_t) total) > 0.06) {
                    model_perf = evaluate(model, 0.15);
                    if (model_perf[1] > best_perf[1]) {
                        model_perf = evaluate(model, .3);
                        if (model_perf[1] > best_perf[1]) {
                            losing_streak = max_streak;
                            best_model = model;
                            best_perf = model_perf;
                            best_epoch = e + 1;
                        } else {
                            losing_streak--;
                        }
                    } else if (losing_streak <= 0) {
                        break;
                    } else {
                        losing_streak--;
                        if (lr > lr_min) {
                            lr /= 1.02;// -= lr_step;
                        } else {
                            //lr = lr_max;
                        }
                    }
                }

            }

            model_perf = evaluate(model, 1);
            if (model_perf[1] > best_perf[1]) {
                best_model = model;
                best_perf = model_perf;
                best_epoch = epochs;
            }
            print_inf("Best Epoch",best_epoch);
            print_accuracy(best_perf);

            num_t total_vars = 0;
            num_t total_zeroes = 0;
            for (auto &l: best_model) {
                total_zeroes += var_get_weights_zeroes(l);
                total_vars += var_get_weights_size(l);
            }
            print_inf("model sparsity",total_zeroes / total_vars,"size:",total_vars);

            auto epoch_time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> diff = epoch_time_end - sgd_timer;

            print_inf("Finished stochastic gradient descent. Taking",diff.count(),"seconds.");
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
        num_t loss(const vec_t &actual, const vec_t &predicted) {
            vec_t r = actual - predicted;
            r = r.array().pow(2);
            return r.mean();
        }

        static inline vec_t loss_prime(const vec_t &actual, const vec_t &predicted) {
            vec_t r = predicted - actual;
            r = 2.f * r / actual.size(); // standard mean squared loss
            return r;
        }

        static void
        update_sample(const vec_t &a0_, const vec_t &target_, size_t batch_index, num_t lr, VarLayers &model) {
            var_start_sample(model);
            vec_t a0 = a0_, target = target_;
            vec_t result = var_feed_forward(a0, model);

            vec_t error = loss_prime(target, result);

            var_bp(error, lr, model);
            var_end_sample(model);
        }

        static void
        update_sample(const training_set &data, size_t batch_index,
                      num_t lr, VarLayers &model) {

            vec_t a0, target;
            a0 = data.training_inputs[batch_index];
            target = data.training_outputs[batch_index];
            update_sample(a0, target, batch_index, lr, model);
        }

        void update_layers(VarLayers &dest, const VarLayers &source) {
            auto isource = source.begin();
            auto idest = dest.begin();
            for (; isource != source.end() && idest != dest.end(); ++isource, ++idest) {
                if (!var_layer_update_bp(*idest, *isource)) {
                    print_err("layer type not found");
                }
            }
        }

        void raw_copy(VarLayers &dest, const VarLayers &source) {
            auto isource = source.begin();
            auto idest = dest.begin();
            for (; isource != source.end() && idest != dest.end(); ++isource, ++idest) {
                if (!var_layer_raw_copy(*idest, *isource)) {
                    print_err("layer type not found");
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
        void update_mini_batch(vector<int> &indices, int batch_num, num_t lr, VarLayers &model) {
            int batch_index = 0;
            var_start_batch(model);
            for (int b = batch_num * mini_batch_size_;
                 b < ((batch_num * mini_batch_size_) + mini_batch_size_) && b < data.training_outputs.size();
                 b++) {
                batch_index = indices[b];
                vec_t a0, target;
                a0 = data.training_inputs[batch_index];
                target = data.training_outputs[batch_index];
                update_sample(a0, target, batch_index, lr, model);

            }
            var_end_batch(model);
            var_update_weights(model, mini_batch_size_); // accumulate all the errors

        }

        void update_mini_batch_single(vector<int> &indices, int batch_num, num_t lr, VarLayers &model) {
            int batch_index = 0;
            vec_t a0, target;
            var_start_batch(model);
            for (int b = batch_num * mini_batch_size_;
                 b < ((batch_num * mini_batch_size_) + mini_batch_size_) && b < data.training_outputs.size();
                 b++) {
                batch_index = indices[b];
                vec_t a0, target;
                a0 = data.training_inputs[batch_index];
                target = data.training_outputs[batch_index];
                update_sample(a0, target, batch_index, lr, model);
            }
            var_end_batch(model);
        }

        void print_training_data() {
            print_inf("Training set in_size", data.training_inputs.size(),
                      "Testing set in_size", data.test_labels.size());
        }

        void print_accuracy(array<num_t, 2> result) {

            print_inf("Accuracy: Train =", result[0] * 100,"%, ",
            "Validation =", result[1] * 100, "%");
        }
        /**
         * evaluate model stochastically
         * @param model the model to evaluate
         * @param fraction of test and training inputs to evaluate, fraction can be larger than 1 if you like
         * @return { test accuracy, training accuracy }
         */
        // stochastic evaluation
        array<num_t, 2> evaluate(VarLayers &model, num_t fraction_ = 1) {
            num_t fraction = abs(fraction_);
            if (fraction > 2) fraction = 1;
            std::random_device rd;
            std::mt19937 g(rd());

            array<num_t, 2> result;
            size_t num_correct = 0;
            size_t output;
            size_t train_output;
            var_set_training(model, false);
            std::uniform_int_distribution<size_t> dis_t(0, data.training_outputs.size() - 1);
            for (uint32_t i = 0; i < data.training_outputs.size() * fraction; i++) {
                size_t o_index = dis_t(g);
                var_feed_forward(data.training_inputs[o_index], model);
                vec_t vi = var_get_input(model.back());
                vi.maxCoeff(&output);
                data.training_outputs[o_index].maxCoeff(&train_output);
                if (output == train_output) {
                    num_correct++;
                }
            }

            std::uniform_int_distribution<size_t> dis(0, data.test_inputs.size() - 1);

            result[0] = (num_t) num_correct / (fraction * data.training_inputs.size());
            num_correct = 0;
            for (uint32_t i = 0; i < data.test_labels.size() * fraction; i++) {
                size_t sample_index = dis(g);
                var_feed_forward(data.test_inputs[sample_index], model);
                var_get_input(model.back()).maxCoeff(&output);
                if (output == data.test_labels[sample_index]) {
                    num_correct++;
                }
            }

            result[1] = (num_t) num_correct / (fraction * data.test_labels.size());

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
                print_inf("Successfully saved to ", filepath, ".");
                return 0;
            }
            print_inf("Failed to save to", filepath, ".");
            return 1;
        }

    };

}

#endif