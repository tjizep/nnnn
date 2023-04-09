//
// Created by Pretorius, Christiaan on 2022-10-01.
//

#ifndef NNNN_ACTIVATIONS_H
#define NNNN_ACTIVATIONS_H

#include <basics.h>
#include <abstract_layer.h>

namespace noodle {
    using namespace std;
    using namespace Eigen;


    struct sigmoid {
        num_t operator()(num_t x) const {
            return 1 / (1 + exp(-x));
        }
    };

    struct sigmoid_derivative {
        num_t operator()(num_t x) const {
            return x * (1 - x);
        }
    };

    struct swish {
        num_t beta = 1;
        sigmoid s;
        swish(num_t b) : beta(b){}
        num_t operator()(num_t x) const {
            return x * s(beta * x);
        }
    };

    struct swish_derivative {
        num_t beta = 1;
        swish_derivative(num_t b) : beta(b){}
        num_t operator()(num_t x) const {
            return (exp(-x)*(x + 1) + 1)/pow((1+exp(-x)),2);
        }
    };

    struct low_sigmoid {
        num_t flatness = 0.77;

        low_sigmoid(num_t flatness) : flatness(flatness) {};

        num_t operator()(num_t x) const {
            return 1 / (1 + exp(-flatness * x));
        }
    };

    struct low_sigmoid_derivative {
        num_t flatness = 0.77;

        low_sigmoid_derivative(num_t flatness) : flatness(flatness) {};

        num_t operator()(num_t x) const {
            return (1 / flatness) * x * (1 - x);
        }
    };

    struct tanh_activation {
        num_t operator()(num_t x) const {
            return (2 / (1 + exp(-2 * x))) - 1;
        }
    };

    struct tanh_derivative {
        num_t operator()(num_t x) const {
            return 1.0f - (x * x);
        }
    };

    struct relum {
        num_t max_val = 1000;

        num_t operator()(num_t x) const {
            return (x >= 0) ? x : x / max_val;
        }
    };

    struct relu_derivative {
        num_t max_val = 1000;
        num_t operator()(num_t x) const {
            return (x >= 0) ? 1 : x / 1000;
        }
    };

    struct sigmoid_layer : public abstract_layer {

        sigmoid_layer() : abstract_layer("SIGMOID") {}

        vec_t forward(const vec_t &io) {
            assert(io.size() > 0);
            return io.unaryExpr(sigmoid{});
        }

        void bp(gradients& state, const vec_t &output_error, num_t /*learning_rate*/) {
            assert(output_error.size() > 0);
            vec_t cd = state.output.unaryExpr(sigmoid_derivative{});
            assert(output.rows() == input.rows());
            state.bp_output = output_error.cwiseProduct(cd);
        }
    };
    struct swish_layer : public abstract_layer {
        num_t beta = 1;

        swish_layer(num_t beta) : abstract_layer("SWISH"), beta(beta) {}
        swish_layer() : abstract_layer("SWISH") {}

        vec_t forward(const vec_t &io) {
            assert(io.size() > 0);
            return io.unaryExpr(swish{beta});
        }

        void bp(gradients& state, const vec_t &output_error, num_t /*learning_rate*/) {
            assert(output_error.size() > 0);
            vec_t cd = state.output.unaryExpr(swish_derivative{beta});
            assert(output.rows() == input.rows());
            state.bp_output = output_error.cwiseProduct(cd);
        }
    };
    struct tanh_layer : public abstract_layer {

        tanh_layer() : abstract_layer("TANH") {}

        vec_t forward(const vec_t &io) {
            assert(io.size() > 0);
            return io.unaryExpr(tanh_activation{});
        }

        void bp(gradients& state, const vec_t &output_error, num_t /*learning_rate*/) {
            assert(output_error.size() > 0);
            vec_t cd = state.output.unaryExpr(tanh_derivative{});
            assert(state.output.rows() == input.rows());
            state.bp_output = output_error.cwiseProduct(cd);
        }
    };

    /**
     * basically a sigmoid activation with a flatness parameter that helps back propagation
     * works about as well as leaky rely but using together with leaky relu after other layers
     * Not directly after leaky relu but instead of relu at the end of a deep network
     * or the last fully connected layer.
     * Can improve model accuracy but wont reduce it. There's probably another name for this somewhere.
     */
    struct low_sigmoid_layer : public abstract_layer {
        num_t flatness = 0.37;

        low_sigmoid_layer() : abstract_layer("LOW_SIGMOID") {}

        low_sigmoid_layer(num_t flatness) : abstract_layer("LOW_SIGMOID"), flatness(flatness) {}

        vec_t forward(const vec_t &io) {
            return io.unaryExpr(low_sigmoid{flatness});
        }

        void bp(gradients& state, const vec_t &output_error, num_t /*learning_rate*/) {
            assert(output_error.size() > 0);
            vec_t cd = state.output.unaryExpr(low_sigmoid_derivative{flatness});
            assert(state.output.rows() == input.rows());
            state.bp_output = output_error.cwiseProduct(cd);

        }
    };

    struct relu_layer : public abstract_layer {

        num_t leakiness = 1; // use factors > 1 i.e. 10, 100, 1000

        relu_layer(num_t leakiness = 1) : abstract_layer("LRELU") {
            this->leakiness = leakiness;
        }

        vec_t forward(const vec_t &io) {
            num_t re = this->leakiness;
            return io.unaryExpr([&](num_t x) -> num_t {
                return x > 0 ? x / re : 0;
            });
        }

        void bp(gradients& state, const vec_t &output_error, num_t /*learning_rate*/) {
            num_t re = this->leakiness;
            vec_t cd = state.output.unaryExpr([&](num_t x) -> num_t {
                return x > 0 ? 1 : 1 / re;
            });
            state.bp_output = output_error.cwiseProduct(cd);
        }

    };

    struct normalize_layer : public abstract_layer {
        vec_t input = row_vector();
        vec_t output = row_vector();

        normalize_layer() : abstract_layer("NORMALIZE") {
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;
            output.normalize();
            return output;
        }

        void bp(gradients& state, const vec_t &output_error, num_t /*learning_rate*/) {
            state.bp_output = output_error; //output_error.cwiseProduct(x);
            state.bp_output *= 2;
        }
    };

    struct dropout_layer : public abstract_layer {
        vec_t input = row_vector();
        vec_t output = row_vector();
        num_t ratio = 0.1;
        bool is_training = true;
        std::default_random_engine generator;

        dropout_layer(num_t ratio) : abstract_layer("DROPOUT") {
            this->ratio = ratio;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;
            if (is_training && ratio > 0) {
                std::binomial_distribution<int> distribution(1, 1 - ratio);
                output = output.unaryExpr([&](num_t x) -> num_t {
                    auto c = distribution(generator);
                    if (c) return x;
                    return 0;
                });
            }

            return output;
        }

        void set_training(bool training) {
            this->is_training = training;
        }

        void bp(gradients& state, const vec_t &output_error, num_t /*learning_rate*/) {
            state.bp_output = output_error;
        }
    };

    struct pepper_layer : public abstract_layer {
        num_t ratio = 1;
        bool is_training = true;
        std::default_random_engine generator;

        pepper_layer(num_t ratio) : abstract_layer("PEPPER") {
            this->ratio = ratio;
        }

        void set_training(bool training) {
            is_training = true;
        }

        vec_t forward(const vec_t &io) {
            if (is_training) {
                std::uniform_real_distribution<num_t> distribution(-1, 1);
                return io.unaryExpr([&](num_t x) -> num_t {
                    auto pepper = distribution(generator);
                    return x + x * ratio * pepper;
                });
            }

            return io;
        }

        void bp(gradients& state, const vec_t &output_error, num_t learning_rate) {
            state.bp_output = output_error;
        }
    };

    struct soft_max_layer : public abstract_layer {
        soft_max_layer() : abstract_layer("SOFTMAX") {
        }

        vec_t forward(const vec_t &io) {
            vec_t output = io;
            /// the simple softmax function as known by everyone
            /// sum of output is 1
            output.array() -= output.array().maxCoeff();
            output = output.array().exp();
            output.array() /= output.sum();
            //output.array() = output.array().log();
            return output;
        }

        void bp(gradients& state, const vec_t &output_error, num_t /*learning_rate*/) {
            assert(output_error.size() > 0);
            assert(output.rows() == input.rows());

#if 0
            vec_t softmax = output;
            mat_t d_softmax = (
                    softmax * vec_t::Identity(softmax.rows(), softmax.cols())
                    - (softmax.transpose() * softmax)).array();
#endif
            state.bp_output = output_error; // TODO: this is not correct but its working anyway
        }
    };

}
#endif //NNNN_ACTIVATIONS_H
