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
#include <graph.h>

namespace noodle {
    using namespace std;
    using namespace Eigen;
    struct training_set {
        string name; /// something to ID this with
        vector<int> training_labels;
        vector<noodle::vec_t> training_inputs ;
        vector<noodle::vec_t> training_outputs ;
        vector<int> test_labels ;
        vector<noodle::vec_t> test_inputs;
        vector<noodle::vec_t> test_outputs;
        int32_t data_size;
    };

    typedef std::array<num_t, 2> LearningRate;

    class trainer {
    public:
    private:
        training_set data;
        size_t mini_batch_size_ = 3;
        LearningRate learning_rate_ = {0.3, 0.01};

    public:


        trainer(const training_set &data, num_t mini_batch_size, std::array<num_t, 2> learning_rate) {
            auto prev = 0;
            this->data = data;
            mini_batch_size_ = mini_batch_size;
            learning_rate_ = std::move(learning_rate);
        }

        static vec_t var_feed_forward(vec_t &a0, graph &model) {
            index_t l = 0;
            graph::forward_selector fwrd = model.first();
            if(!fwrd.ok(model)){
                fatal_err("graph seems empty");
                return a0;
            }
            index_t last = fwrd.get();
            fwrd.set_activation(model, a0);
            for (;fwrd.ok(model);fwrd.next(model)) {
                if(!fwrd.forward(model)){
                    return a0;
                }
                last = fwrd.get();
            }
            return model.resolve(last).output;
        }

        static inline void var_bp(const vec_t &error, num_t lr, graph &model) {
            print_dbg("var_bp graph", error.size());
            graph::reverse_selector bw = model.last();
            if(!bw.ok(model)){
                fatal_err("graph seems empty");
                return;
            }
            index_t l = 0;
            bw.set_error(model, error);
            while(bw.ok(model)){
                print_dbg("var_bp layer", l++, bw.resolve(model).name, var_get_name(bw.resolve(model).operation));
                //error = var_layer_bp(bw.resolve(model).operation, error, lr);
                if(!bw.backward(model, lr)){
                    return;
                }
                bw.next(model);
            }
        }

        enum{
            TRAIN = 0,
            TEST = 1,
            BOTH
        };
        struct model_perf_t{
            num_t test{0};
            num_t train{0};
            model_perf_t(){

            }
            model_perf_t(num_t test, num_t train) : test(test), train(train){

            }
        };
        template<typename ModelType>
        ModelType
        stochastic_gradient_descent(ModelType &model, uint32_t epochs, size_t shards = 1, num_t max_streak = 3) {
            ModelType best_model;
            model_perf_t model_perf, best_perf;
            /// NB: we can never use model_perf[0] or best_perf[0]
            /// because then we are using test accuracy during training

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
                    vector <ModelType > tmod(shards);
                    for (auto &tm: tmod)
                        tm = model;
                    for (size_t t = 0; t < shards; ++t) {
                        threads[t] = thread([&](size_t at) {
                            size_t buffer = 0;
                            for (index_t batch_num = 0;
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
                    for (index_t batch_num = 0; batch_num * mini_batch_size_ < data.training_inputs.size(); batch_num++) {
                        update_mini_batch(indices, batch_num, lr, model);
                        ++ix;
                    }
                }
                auto epoch_time_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> diff = epoch_time_end - epoch_timer;
                print_inf("Completing Epoch",e,". complete ",
                     100 * ix / total, "%");
                print_inf("estimated acc.:",best_perf.test);
                print_inf("last epoch duration:",diff.count(),"s, lr:",lr);
                if (((num_t) ix / (num_t) total) > 0.06) {
                    model_perf = evaluate(model, 0.15, TEST);
                    if (model_perf.test > best_perf.test) {
                        model_perf = evaluate(model, .3, TEST);
                        if (model_perf.test > best_perf.test) {
                            losing_streak = max_streak;
                            best_model = model;
                            best_perf = model_perf;
                            best_epoch = e + 1;

                        } else {
                            losing_streak--;
                        }
                    } else if (max_streak > 0 && losing_streak <= 0) {
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
            if(best_model.empty()){
                best_model = model;
                best_perf = evaluate(best_model, 1, TEST);
            }else{
                model_perf = evaluate(model, 1, TEST);
                best_perf = evaluate(best_model, 1, TEST);
                if (model_perf.test > best_perf.test) {
                    best_model = model;
                    best_perf = model_perf;
                    best_epoch = epochs;
                }
            }
            print_dbg("Best estimated Epoch",best_epoch);
            //print_accuracy(best_perf);

            num_t total_vars = 0;
            num_t total_zeroes = 0;
            for (auto &l: get_iterable(best_model)) {
                total_zeroes += var_get_weights_zeroes(get_layer(l));
                total_vars += var_get_weights_size(get_layer(l));
            }
            print_inf("model sparsity",total_zeroes / total_vars,"size:",total_vars);

            auto epoch_time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> diff = epoch_time_end - sgd_timer;

            print_inf("Finished stochastic gradient descent. Taking",diff.count(),"seconds.");
            return best_model;
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
        update_sample(const vec_t &a0_, const vec_t &target_, size_t batch_index, num_t lr, graph &model) {
            print_dbg("update_sample_layers", batch_index);
            model.start_sample();
            vec_t a0 = a0_, target = target_;
            vec_t result = var_feed_forward(a0, model);

            vec_t error = loss_prime(target, result);

            var_bp(error, lr, model);
            model.end_sample();
        }

        void update_layers(graph &dest, const graph &source) {
            source.update_layers(dest);
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
         template<typename ModelType>
        void update_mini_batch(vector<int> &indices, int batch_num, num_t lr, ModelType &model) {
            int batch_index = 0;
            print_dbg("update_mini_batch",batch_num);
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

        template<typename ModelType>
        void update_mini_batch_single(vector<int> &indices, int batch_num, num_t lr, ModelType &model) {
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

        void print_accuracy(model_perf_t result) {

            print_inf("Accuracy: Train =", result.train * 100,"%, ",
            "Validation =", result.test * 100, "%");
        }
        /**
         * evaluate model stochastically
         * @param model the model to evaluate
         * @param fraction of test and training inputs to evaluate, fraction can be larger than 1 if you like
         * @return { test accuracy, training accuracy }
         */
        // stochastic evaluation
        template<typename ModelType>
        model_perf_t evaluate(ModelType &model, num_t fraction_ = 1, index_t which = BOTH) {
            num_t fraction = abs(fraction_);
            if (fraction > 2) fraction = 1;
            std::random_device rd;
            std::mt19937 g(rd());

            model_perf_t result;
            size_t num_correct = 0;
            size_t output;
            size_t train_output;
            var_set_training(model, false);
            if(which == TRAIN||which == BOTH) {
                std::uniform_int_distribution<size_t> dis_t(0, data.training_outputs.size() - 1);
                for (uint32_t i = 0; i < data.training_outputs.size() * fraction; i++) {
                    size_t o_index = dis_t(g);
                    vec_t vi = var_feed_forward(data.training_inputs[o_index], model);

                    vi.maxCoeff(&output);
                    data.training_outputs[o_index].maxCoeff(&train_output);
                    if (output == train_output) {
                        num_correct++;
                    }
                }
                result.train = (num_t) num_correct / (fraction * data.training_inputs.size());
            }else{
                result.train = 0;
            }

            std::uniform_int_distribution<size_t> dis(0, data.test_inputs.size() - 1);
            if(which==TEST||which==BOTH) {

                num_correct = 0;
                for (uint32_t i = 0; i < data.test_labels.size() * fraction; i++) {
                    size_t sample_index = dis(g);
                    vec_t back = var_feed_forward(data.test_inputs[sample_index], model);
                    back.maxCoeff(&output);
                    if (output == data.test_labels[sample_index]) {
                        num_correct++;
                    }
                }

                result.test = (num_t) num_correct / (fraction * data.test_labels.size());
            }else{
                result.test = 0;
            }
            var_set_training(model, true);
            return result;
        }
    };

}

#endif