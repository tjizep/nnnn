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
        vec_t input = row_vector();
        vec_t output = row_vector();

        sigmoid_layer() : abstract_layer("SIGMOID") {}

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        vec_t forward(const vec_t &io) {
            assert(io.size() > 0);
            input = io;
            //cout << "L " << depth << " " << __FUNCTION__ << " " << name << " " << input.rows() << " input " << input.norm() << endl;
            output = io.unaryExpr(sigmoid{});
            return output;
        }

        vec_t bp(const vec_t &output_error, num_t /*learning_rate*/) {
            assert(output_error.size() > 0);
            vec_t cd = output.unaryExpr(sigmoid_derivative{});
            //cout << "L " << depth << " " << __FUNCTION__ << " " << name << " " << input.size() << " err " << output_error.norm() << " cder " << cd.norm() << endl;
            assert(output.rows() == input.rows());
            vec_t result = output_error.cwiseProduct(cd);
            return result;
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
        vec_t input = row_vector();
        vec_t output = row_vector();
        num_t flatness = 0.37;

        low_sigmoid_layer() : abstract_layer("LOW_SIGMOID") {}

        low_sigmoid_layer(num_t flatness) : abstract_layer("LOW_SIGMOID"), flatness(flatness) {}

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input.unaryExpr(low_sigmoid{flatness});

            return output;
        }

        vec_t bp(const vec_t &output_error, num_t /*learning_rate*/) {
            assert(output_error.size() > 0);
            vec_t cd = output.unaryExpr(low_sigmoid_derivative{flatness});
            //cout << "L " << depth << " " << __FUNCTION__ << " " << name << " " << input.size() << " err " << output_error.norm() << " cder " << cd.norm() << endl;
            assert(output.rows() == input.rows());
            vec_t result = output_error.cwiseProduct(cd);

            return result;
        }
    };

    struct relu_layer : public abstract_layer {

        vec_t input = row_vector();
        vec_t output = row_vector();
        num_t leakiness = 1; // use factors > 1 i.e. 10, 100, 1000

        relu_layer(num_t leakiness = 1) : abstract_layer("LRELU") {
            this->leakiness = leakiness;
        }

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;
            num_t re = this->leakiness;
            output = output.unaryExpr([&](num_t x) -> num_t {
                return x > 0 ? x / re : 0;
            });
            return output;
        }

        vec_t bp(const vec_t &output_error, num_t learning_rate) {
            num_t re = this->leakiness;
            vec_t cd = input.unaryExpr([&](num_t x) -> num_t {
                return x > 0 ? 1 : 1 / re;
            });

            vec_t result = output_error.cwiseProduct(cd);

            return result;
        }

    };

    struct normalize_layer : public abstract_layer {
        vec_t input = row_vector();
        vec_t output = row_vector();

        normalize_layer() : abstract_layer("NORMALIZE") {
        }

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;
            output.normalize();
            return output;
        }

        vec_t bp(const vec_t &output_error, num_t learning_rate) {
            vec_t result = output_error; //output_error.cwiseProduct(x);
            result *= 2;
            return result;
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

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;
            if (is_training) {
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

        vec_t bp(const vec_t &output_error, num_t learning_rate) {
            return output_error;
        }
    };

    struct pepper_layer : public abstract_layer {
        vec_t input = row_vector();
        vec_t output = row_vector();
        num_t ratio = 1;
        bool is_training = true;
        std::default_random_engine generator;

        pepper_layer(num_t ratio) : abstract_layer("PEPPER") {
            this->ratio = ratio;
        }

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        void set_training(bool training) {
            is_training = true;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;
            if (is_training) {
                std::uniform_real_distribution<num_t> distribution(-1, 1);
                output = output.unaryExpr([&](num_t x) -> num_t {
                    auto pepper = distribution(generator);
                    return x + x * ratio * pepper;
                });
            }

            return output;
        }

        vec_t bp(const vec_t &output_error, num_t learning_rate) {
            std::uniform_real_distribution<num_t> distribution(-1, 1);
            vec_t output = output_error.unaryExpr([&](num_t x) -> num_t {
                auto pepper = distribution(generator);
                return x + x * ratio * pepper;
            });
            return output_error;
        }
    };

    struct soft_max_layer : public abstract_layer {
        vec_t input = row_vector();
        vec_t output = row_vector();

        soft_max_layer() : abstract_layer("SOFTMAX") {
        }

        soft_max_layer(const soft_max_layer &right) : abstract_layer("SOFTMAX") {
            *this = right;
        }

        soft_max_layer &operator=(const soft_max_layer &right) {
            input = right.input;
            output = right.output;
            return *this;
        }

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            output = input;
            /// the simple softmax function as known by everyone
            /// sum of output is 1
            output.array() -= output.array().maxCoeff();
            output = output.array().exp();
            output.array() /= output.sum();
            //output.array() = output.array().log();
            return output;
        }

        vec_t bp(const vec_t &output_error, num_t learning_rate) {
            assert(output_error.size() > 0);
            assert(output.rows() == input.rows());

#if 0
            vec_t softmax = output;
            mat_t d_softmax = (
                    softmax * vec_t::Identity(softmax.rows(), softmax.cols())
                    - (softmax.transpose() * softmax)).array();
#endif
            vec_t result;
            result = output_error; // TODO: this is not correct but its working anyway
            return result;
        }
    };

}
#endif //NNNN_ACTIVATIONS_H
