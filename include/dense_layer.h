//
// Created by Pretorius, Christiaan on 2022-10-01.
//

#ifndef NNNN_DENSE_LAYER_H
#define NNNN_DENSE_LAYER_H

#include <basics.h>
#include <message.h>
#include <sparsity.h>
#include <activations.h>

namespace noodle {
    using namespace std;
    using namespace Eigen;

    struct fc_layer : public abstract_layer {
        uint32_t in_size = 0;
        uint32_t out_size = 0;
        index_t rounding = 5;
        block_sparsity sparseness;
        num_t momentum = 0;
        vec_t biases = row_vector();
        mat_t weights = matrix();

        vec_t output = row_vector();
        /// temp data during training
        vec_t input_error = row_vector();
#if 0
        mat_t mini_batch_update_weights = matrix();
        vec_t mini_batch_update_biases = row_vector();
#endif
        fc_layer(uint32_t in_size, uint32_t out_size, num_t sparseness = 0, num_t sparsity_greed = 2.5, num_t momentum = 0) : abstract_layer(
                "FULLY CONNECTED") {
            print_dbg("name", name);
            print_dbg("in_size",in_size);
            print_dbg("out_size",out_size);
            this->in_size = in_size;
            this->out_size = out_size;
            this->momentum = momentum;
            this->sparseness.sparseness = std::min<num_t>(0.99, abs(sparseness));
            this->sparseness.sparsity_greediness = std::max<num_t>(1.01, sparsity_greed);
            initialize();

        }

        const mat_t &get_weights() const {
            return weights;
        }
        void round_(){
            round(weights,rounding);
            round(biases,rounding);
        }
        bool initialize() {

            num_t mc;
            if(biases.size()==0)
                biases = vec_t::Random(out_size);
            //biases.array() /= biases.array().abs().maxCoeff();
            //biases.array() -= 0.5;
            if(weights.size()==0)
                weights = mat_t::Random(out_size, in_size);
            //weights.array() /= weights.array().abs().maxCoeff();
            //weights.array() -= 0.5;
            round_();
            return true;
        }

        void get_message(message& m) const {

            m.data["weights"] = this->weights;
            m.data["biases"] = this->biases;
            m.kind = name;
        }

        void put_message(const message& m)  {

            if(m.kind != name){
                fatal_err("invalid 'kind'");
            }

            if(!m.data.contains("weights")) {
                fatal_err("'weights' not found");
            }

            if(!m.data.contains("biases")){
                fatal_err("'biases' not found");
            }

            auto vw = m.data.find("weights");
            if(const mat_t* w= std::get_if<mat_t>(&vw->second)){
                this->weights = *w;
            }else{
                fatal_err("invalid 'weights'");
            }

            auto vb = m.data.find("biases");
            if(const vec_t* b= std::get_if<vec_t>(&vb->second)){
                this->biases = *b;
            }else{
                fatal_err("invalid 'biases' found");
            }
        }

        vec_t forward(const vec_t &input) {
            //output = weights * input;
            sparseness.vec_mul_assign(output, weights, input);
            assert(output.rows() == out_size);
            assert(biases.size() == out_size);
            output += biases;

            return output;
        }

        num_t get_weights_sparseness() const {
            return sparseness.get_sparseness(weights);
        }

        num_t get_weights_zeroes() const {
            return sparseness.get_zeroes(weights);
        }

        void start_batch() {
        }

        void end_batch() {
        }

        void update_batch_variables(message& variables) {
            if (variables.get_mat("mini_batch_update_weights").size() == 0) return;
            //print_dbg("batch update weights",variables.get_mat("mini_batch_update_weights").array().sum());
            weights += variables.get_mat("mini_batch_update_weights");
            biases += variables.get_vector("mini_batch_update_biases");
            if (sparseness.sparseness && variables.get_number("train_percent") > 0.05)
                sparseness.reduce_weights(weights);

            round_();

            variables.get_mat("mini_batch_update_weights") = matrix();
            variables.get_vector("mini_batch_update_biases") = row_vector();

        }


        void update_variables(message& dest_variables, const fc_layer &fc, const message& src_variables) {
            if (src_variables.get_mat("mini_batch_update_weights").size() > 0) {
                assign_add(dest_variables.get_mat("mini_batch_update_weights"), src_variables.get_mat("mini_batch_update_weights"));
            }
            if (src_variables.get_vector("mini_batch_update_biases").size() > 0) {
                assign_add(dest_variables.get_vector("mini_batch_update_biases"), src_variables.get_vector("mini_batch_update_biases"));
            }
        }

        void raw_copy_from(const fc_layer &fc) {
            // *this = fc;

            if (weights.size() > 0 && fc.weights.size() == weights.size()) {
                //memcpy(&weights(0), &fc.weights(0), sizeof(num_t) * weights.size());
                weights = fc.weights;
            } else {
                weights = fc.weights;
            }
            if (biases.size() > 0 && fc.biases.size() == biases.size()) {
                //memcpy(&biases(0), &fc.biases(0), sizeof(num_t) * biases.size());
                //biases = fc.biases;
            } else {
                biases = fc.biases;
            }
            in_size = fc.in_size;
            out_size = fc.out_size;
            //index = fc.index;
            input_error = fc.input_error;
            output = fc.output;
            momentum = 0;
            sparseness = fc.sparseness;
            //mini_batch_update_weights.array() = 0;// = fc.mini_batch_update_weights;
            //mini_batch_update_biases.array() = 0;// = fc.mini_batch_update_biases;
            round_();

        }

        __attribute__((noinline))
        void update_mini_batch_weights(gradients& state, num_t learning_rate, const vec_t output_error) {

            vec_t b_delta;

            b_delta = -learning_rate * output_error;
            assign_add(state.variables.get_vector("mini_batch_update_biases"), b_delta);
        }

        /// output error is from next layer below this one (since its reverse prop) or start
        void bp(gradients& state, gradients& shared, const vec_t &output_error) {

            assert(out_size == 0 || out_size == output_error.rows());
            assert(weights.size() > 0);
            num_t learning_rate = shared.variables.get_number("learning_rate");

            //weights_error = output_error * input.transpose();
            //sparseness.project_mul(weights_error, output_error, input, -learning_rate);
            /// fusion operation into mini_batch_update_weights
            sparseness.project_mul_add(state.variables.get_mat("mini_batch_update_weights"), output_error,
                                       state.activation, -learning_rate);

            update_mini_batch_weights(state, learning_rate, output_error);
            index_t index = 1;
            if (index > 0) {

                //input_error = weights.transpose() * output_error; // don't calculate this on the top layer
                // 100x50 * 50x1 = 100x1
                //state.bp_output
                sparseness.mask_mul(state.bp_output, weights, output_error);

            } else {
                state.bp_output = output_error;
            }

        }

    };


}
#endif //NNNN_DENSE_LAYER_H
