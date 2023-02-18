//
// Created by kriso on 2/15/2023.
//

#ifndef NNNN_ENSEMBLE_H
#define NNNN_ENSEMBLE_H
#include <basics.h>
#include <abstract_layer.h>

namespace noodle {
    using namespace std;
    using namespace Eigen;
    template<typename LayersForward>
    struct ensemble : public abstract_layer {
        uint32_t in_size = 0;
        uint32_t out_size = 0;
        uint32_t index = 0;
        vec_t input = row_vector();
        vec_t output = row_vector();
        /// temp data during training
        vec_t input_error = row_vector();
        LayersForward layers;
        ensemble(const LayersForward &layers,uint32_t in_size, uint32_t out_size) : abstract_layer(
                "ENSEMBLE") {
            this->in_size = in_size;
            this->out_size = out_size;
            this->layers = layers;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            //output = weights * input;
            output = layers.feed_forward(io);

            return output;
        }

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        num_t get_weights_sparseness() const {
            return 0.0f;
        }

        num_t get_weights_zeroes() const {
            //return sparseness.get_zeroes(weights);
            return 0.0f;
        }

        void update_weights(const num_t train_percent) {
        }

        /***
         * called when shards need to update the origin model
         * not thread safe so latches/locks should be takem
         * @param fc the shard
         */
        void update_bp_from(const ensemble &fc) {

        }

        void raw_copy_from(const ensemble &fc) {
            // *this = fc;

        }

        __attribute__((noinline))
        void update_mini_batch_weights(num_t learning_rate, const vec_t output_error) {

        }

        /// output error is from next layer below this one (since its reverse prop) or start
        vec_t bp(const vec_t &output_error, num_t learning_rate) {

            assert(out_size == 0 || out_size == output_error.rows());
            assert(weights.size() > 0);
            input_error = layers.back_prop(output_error,learning_rate);
            return input_error;

        };
    };
}
#endif //NNNN_ENSEMBLE_H
