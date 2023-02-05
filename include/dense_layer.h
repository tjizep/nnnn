//
// Created by Pretorius, Christiaan on 2022-10-01.
//

#ifndef NNNN_DENSE_LAYER_H
#define NNNN_DENSE_LAYER_H

#include <basics.h>

#include <sparsity.h>
#include <activations.h>
#include <optimizers.h>
#include <type_traits>
#include <variant>
#include <type_traits>

namespace noodle {
    using namespace std;
    using namespace Eigen;

    struct abstract_layer {
        string name;

        abstract_layer(string name) : name(name) {};
    };


    struct fc_layer : public abstract_layer {
        uint32_t in_size = 0;
        uint32_t out_size = 0;
        uint32_t index = 0;
        block_sparsity sparseness;
        num_t momentum = 0;
        vec_t biases = NULL_V();
        mat_t weights = NULL_M();

        mat_t prev_weights_delta = NULL_M();
        vec_t prev_biases_delta = NULL_V();

        vec_t input = NULL_V();
        vec_t output = NULL_V();
        /// temp data during training
        vec_t input_error = NULL_V();
        mat_t mini_batch_update_weights = NULL_M();
        vec_t mini_batch_update_biases = NULL_V();

        fc_layer(uint32_t in_size, uint32_t out_size, num_t sparseness = 0, num_t momentum = 0) : abstract_layer(
                "FULLY CONNECTED") {

            this->in_size = in_size;
            this->out_size = out_size;
            this->momentum = momentum;
            this->sparseness.sparseness = std::min<num_t>(0.89, abs(sparseness));
            initialize_weights();

        }

        const mat_t &get_weights() const {
            return weights;
        }

        void initialize_weights() {

            num_t mc;
            biases = vec_t::Random(out_size);
            biases.array() /= biases.array().abs().maxCoeff();
            //biases.array() -= 0.5;

            weights = mat_t::Random(out_size, in_size);
            weights.array() /= weights.array().abs().maxCoeff();
            //weights.array() -= 0.5;
        }

        vec_t forward(const vec_t &io) {
            input = io;
            //output = weights * input;
            sparseness.vec_mul_assign(output, weights, input);
            assert(output.rows() == out_size);
            assert(biases.size() == out_size);
            output += biases;

            return output;
        }

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }
        num_t get_weights_sparseness() const {
            return sparseness.get_sparseness(weights);
        }
        num_t get_weights_zeroes() const {
            return sparseness.get_zeroes(weights);
        }

        void start_batch(){
        }

        void end_batch(){
        }


        void update_weights(const num_t train_percent) {
            if(mini_batch_update_weights.size()==0) return;

            weights += mini_batch_update_weights;
            biases += mini_batch_update_biases;

            if (sparseness.sparseness && train_percent > 0.05)
                sparseness.reduce_weights(weights, this->index);
            if (momentum > 0) {
                if (prev_weights_delta.size() > 0)
                    weights += momentum * prev_weights_delta;
                if (prev_biases_delta.size() > 0)
                    biases += momentum * prev_biases_delta;

                prev_weights_delta = mini_batch_update_weights;
                prev_biases_delta = mini_batch_update_biases;
            }
            mini_batch_update_weights = NULL_M();
            mini_batch_update_biases = NULL_V();
        }

        /***
         * called when shards need to update the origin model
         * not thread safe so latches/locks should be takem
         * @param fc the shard
         */
        void update_bp_from(const fc_layer &fc) {
            if (fc.mini_batch_update_weights.size() > 0) {
                assign_add(mini_batch_update_weights, fc.mini_batch_update_weights);
            }
            if (fc.mini_batch_update_biases.size() > 0) {
                assign_add(mini_batch_update_biases, fc.mini_batch_update_biases);
            }
        }
        void raw_copy_from(const fc_layer &fc) {
           // *this = fc;

            if(weights.size() > 0 && fc.weights.size() == weights.size()){
                //memcpy(&weights(0), &fc.weights(0), sizeof(num_t) * weights.size());
                weights = fc.weights;
            }else{
                weights = fc.weights;
            }
            if(biases.size() > 0 && fc.biases.size() == biases.size()){
                //memcpy(&biases(0), &fc.biases(0), sizeof(num_t) * biases.size());
                //biases = fc.biases;
            }else{
                biases = fc.biases;
            }
            in_size = fc.in_size;
            out_size = fc.out_size;
            index = fc.index;
            input = fc.input;
            input_error = fc.input_error;
            output = fc.output;
            momentum = 0;
            sparseness = fc.sparseness;
            mini_batch_update_weights = fc.mini_batch_update_weights;
            mini_batch_update_biases = fc.mini_batch_update_biases;

        }
        __attribute__((noinline))
        void update_mini_batch_weights(num_t learning_rate, const vec_t output_error){

            vec_t b_delta;

            b_delta = -learning_rate * output_error;

            //assign_add(mini_batch_update_weights, weights_error);
            assign_add(mini_batch_update_biases, b_delta);
        }
        /// output error is from next layer below this one (since its reverse prop) or start
        vec_t bp(const vec_t &output_error, num_t learning_rate) {


            assert(out_size == 0 || out_size == output_error.rows());
            assert(weights.size() > 0);

            //weights_error = output_error * input.transpose();
            //sparseness.project_mul(weights_error, output_error, input, -learning_rate);
            sparseness.project_mul_add(mini_batch_update_weights, output_error, input, -learning_rate);

            update_mini_batch_weights(learning_rate, output_error);

            if (index > 0) {

                //input_error = weights.transpose() * output_error; // don't calculate this on the top layer
                // 100x50 * 50x1 = 100x1
                sparseness.mask_mul(input_error, weights, output_error);

            } else {
                input_error = output_error;
            }

            return input_error;
        }

    };

    struct sigmoid_layer : public abstract_layer {
        vec_t input = NULL_V();
        vec_t output = NULL_V();

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
        vec_t input = NULL_V();
        vec_t output = NULL_V();
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

        vec_t input = NULL_V();
        vec_t output = NULL_V();
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
        vec_t input = NULL_V();
        vec_t output = NULL_V();

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
        vec_t input = NULL_V();
        vec_t output = NULL_V();
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
        vec_t input = NULL_V();
        vec_t output = NULL_V();
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
        vec_t input = NULL_V();
        vec_t output = NULL_V();

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

    /**
     * used for optional function checking
     * @tparam T type that has member function
     * @tparam F the member function input type
     * @param f the member function itself
     * @return constexpr true if such a member func exists and thereby enabling further compilation
     */
    template<typename T, typename F>
    constexpr auto has_member_impl(F &&f) -> decltype(f(std::declval<T>()), true) {
        return true;
    }

    template<typename>
    constexpr bool has_member_impl(...) { return false; }

#define has_member(T, EXPR) \
 has_member_impl<T>( [](auto&& obj)->decltype(obj.EXPR){} )

    /**
     * we don't know what youre thinking but this class encapsulates a layer in a model perhaps lateron a node in a graph
     * no virtual abstract methods and all their issues required
     * some methods are optional
     * @tparam V the concrete implementation type
     */
    template<typename V>
    struct model_member {
        typedef V concrete_layer_type;
        V impl;
        uint32_t depth = 0;

        model_member(const V &v) : impl(v) {};

        model_member(const model_member &right) : impl(right.impl) {
        }

        model_member &operator=(const model_member &right) {
            impl = right.impl;
            depth = right.depth;
            return *this;
        }

        vec_t forward(const vec_t &input) {
            return impl.forward(input);
        }

        void set_depth(uint32_t d) {
            depth = d;
        }

        uint32_t get_depth() const {
            return depth;
        }

        /**
         * optional function
         * @param mini_batch_size
         */
        void update_weights(num_t train_percent) {
            if constexpr(has_member(V, update_weights(0)))
                impl.update_weights(train_percent);
        }

        /**
         * optional function
         * @param source
         * @return
         */
        bool update_bp_from(const model_member<V> &source) {
            if constexpr(has_member(V, update_bp_from(source.impl)))
                impl.update_bp_from(source.impl);
            return true;
        }
        bool raw_copy_from(const  model_member<V> &source) {
            if constexpr(has_member(V, raw_copy_from(source.impl)))
                impl.raw_copy_from(source.impl);
            return true;
        }
        /**
         * send back propagation to concrete impl
         * @param input_error
         * @param learning_rate
         * @return the result error to send to layers above (with lower layer depth)
         */
        vec_t bp(const vec_t &input_error, num_t learning_rate) {
            return impl.bp(input_error, learning_rate);
        }

        vec_t get_input() const {
            return impl.get_input();
        }

        void set_training(bool training) {
            if constexpr(has_member(V, set_training(true)))
                return impl.set_training(training);
        }

        num_t get_weights_norm() const {
            if constexpr(has_member(V, get_weights()))
                return impl.get_weights().norm();
            return 0;
        }

        num_t get_weights_zeroes() const {
            if constexpr(has_member(V, get_weights_sparseness())) {
                return impl.get_weights_zeroes();
            }
            return 0;
        }
        num_t get_weights_size() const {
            if constexpr(has_member(V, get_weights_sparseness())) {
                return impl.get_weights().size();
            }
            return 0;
        }

        void start_sample() {
            if constexpr(has_member(V, start_sample())) {
                impl.start_sample();
            }
        }

        void end_sample() {
            if constexpr(has_member(V, end_sample())) {
                impl.end_sample();
            }
        }

        void start_batch() {
            if constexpr(has_member(V, start_batch())) {
                impl.start_batch();
            }
        }

        void end_batch() {
            if constexpr(has_member(V, end_batch())) {
                impl.end_batch();
            }
        }

        string get_name() const {
            return impl.name;
        }
    };

    typedef std::variant<
            model_member<fc_layer>,
            model_member<sigmoid_layer>,
            model_member<low_sigmoid_layer>,
            model_member<relu_layer>,
            model_member<soft_max_layer>,
            model_member<normalize_layer>,
            model_member<dropout_layer>,
            model_member<pepper_layer>> layer;

    typedef std::vector<layer> VarLayers;

    void var_initialize(VarLayers &model) {
        uint32_t d = 0, ix = 0;
        for (auto &m: model) {
            if (model_member<sigmoid_layer> *l = std::get_if<model_member<sigmoid_layer>>(&m)) {
                l->set_depth(d++);
            }
            if (model_member<fc_layer> *l = std::get_if<model_member<fc_layer>>(&m)) {
                l->impl.initialize_weights();
                l->set_depth(d++);
                l->impl.index = ix++;
            }
        }
    }

    void var_set_training(VarLayers &model, bool training) {
        for (auto &m: model) {
            std::visit([&](auto &&arg) {
                arg.set_training(training);
            }, m);
        }
    }

    void var_update_weights(VarLayers &model, num_t train_percent) {
        for (auto &m: model) {
            std::visit([&](auto &&arg) {
                arg.update_weights(train_percent);
            }, m);
        }
    }

    void var_start_batch(VarLayers &model) {
        for (auto &m: model) {
            std::visit([&](auto &&arg) {
                arg.start_batch();
            }, m);
        }
    }

    void var_end_batch(VarLayers &model) {
        for (auto &m: model) {
            std::visit([&](auto &&arg) {
                arg.end_batch();
            }, m);
        }
    }

    void var_start_sample(VarLayers &model) {
        for (auto &m: model) {
            std::visit([&](auto &&arg) {
                arg.start_sample();
            }, m);
        }
    }

    void var_end_sample(VarLayers &model) {
        for (auto &m: model) {
            std::visit([&](auto &&arg) {
                arg.end_sample();
            }, m);
        }
    }

    vec_t var_forward(layer &v, const vec_t &input) {
        return std::visit([&](auto &&arg) -> vec_t {
            return arg.forward(input);
        }, v);
    }

    bool var_layer_update_bp(layer &dest, const layer &source) {
        std::visit([&](auto &&arg) {
            typedef decltype(arg.impl) concrete_layer_type;

            if (const model_member<concrete_layer_type> *src = std::get_if<model_member<concrete_layer_type>>(
                    &source)) {
                arg.update_bp_from(*src);
            }
        }, dest);

        return true;
    }


    bool var_layer_raw_copy(layer &dest, const layer &source) {
        std::visit([&](auto &&arg) {
            typedef decltype(arg.impl) concrete_layer_type;

            if (const model_member<concrete_layer_type> *src = std::get_if<model_member<concrete_layer_type>>(
                    &source)) {
                arg.raw_copy_from(*src);
            }
        }, dest);

        return true;
    }

    static inline vec_t var_layer_bp(layer &v, const vec_t &input_error, num_t learning_rate) {
        return std::visit([&](auto &&arg) -> vec_t {
            return arg.bp(input_error, learning_rate);
        }, v);
    }

    vec_t var_get_input(layer &v) {
        return std::visit([](auto &&arg) -> vec_t {
            return arg.get_input();
        }, v);
    }

    num_t var_get_weights_zeroes(const layer &v) {
        return std::visit([](auto &&arg) -> num_t {
            return arg.get_weights_zeroes();
        }, v);
    }
    num_t var_get_weights_size(const layer &v) {
        return std::visit([](auto &&arg) -> num_t {
            return arg.get_weights_size();
        }, v);
    }

    num_t var_get_weights_norm(const layer &v) {
        return std::visit([](auto &&arg) -> num_t {
            return arg.get_weights_norm();
        }, v);
    }

    string var_get_name(const layer &v) {
        return std::visit([](auto &&arg) -> string {
            return arg.get_name();
        }, v);
    }
}
#endif //NNNN_DENSE_LAYER_H
