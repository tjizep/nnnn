//
// Created by kriso on 2/9/2023.
//

#ifndef NNNN_MODEL_H
#define NNNN_MODEL_H

#include <basics.h>
#include <activations.h>
#include <dense_layer.h>
#include <ensemble.h>
#include <unordered_set>
#include <unordered_map>
#include <list>

namespace noodle {
    using namespace std;

    struct empty_layer : public abstract_layer {
        vec_t biases = row_vector();
        mat_t weights = matrix();
        vec_t input = row_vector();
        vec_t output = row_vector();
        vec_t gradient = row_vector();

        empty_layer() : abstract_layer(
                "EMPTY") {
        }

        const mat_t &get_weights() const {
            return weights;
        }

        vec_t forward(const vec_t &io) {
            output = io;
            input = io;
            return output;
        }

        const vec_t &get_input() const {
            return input;
        }

        vec_t &get_input() {
            return input;
        }



        vec_t bp(const vec_t &output_error, num_t learning_rate) {
            gradient = output_error;
            return gradient;
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
        bool initialize(){
            bool r = true;
            if constexpr (has_member(V, initialize()))
                r = impl.initialize();
            return r;
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
            if constexpr (has_member(V, update_weights(0)))
                impl.update_weights(train_percent);
        }

        /**
         * optional function
         * @param source
         * @return
         */
        bool update_bp_from(const model_member<V> &source) {
            if constexpr (has_member(V, update_bp_from(source.impl)))
                impl.update_bp_from(source.impl);
            return true;
        }

        bool raw_copy_from(const model_member<V> &source) {
            if constexpr (has_member(V, raw_copy_from(source.impl)))
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
            if constexpr (has_member(V, set_training(true)))
                return impl.set_training(training);
        }

        num_t get_weights_norm() const {
            if constexpr (has_member(V, get_weights()))
                return impl.get_weights().norm();
            return 0;
        }

        num_t get_weights_zeroes() const {
            if constexpr (has_member(V, get_weights_sparseness())) {
                return impl.get_weights_zeroes();
            }
            return 0;
        }

        num_t get_weights_size() const {
            if constexpr (has_member(V, get_weights())) {
                return impl.get_weights().size();
            }
            return 0;
        }

        void start_sample() {
            if constexpr (has_member(V, start_sample())) {
                impl.start_sample();
            }
        }

        void end_sample() {
            if constexpr (has_member(V, end_sample())) {
                impl.end_sample();
            }
        }

        void start_batch() {
            if constexpr (has_member(V, start_batch())) {
                impl.start_batch();
            }
        }

        void end_batch() {
            if constexpr (has_member(V, end_batch())) {
                impl.end_batch();
            }
        }

        string get_name() const {
            return impl.name;
        }
    };

    class layer_holder;

    typedef std::variant<
            model_member<empty_layer>,
            model_member<fc_layer>,
            model_member<sigmoid_layer>,
            model_member<low_sigmoid_layer>,
            model_member<tanh_layer>,
            model_member<relu_layer>,
            model_member<soft_max_layer>,
            model_member<normalize_layer>,
            model_member<dropout_layer>,
            model_member<pepper_layer>,
            model_member<ensemble<layer_holder>>> layer;

    typedef std::vector<layer> VarLayers;

    class layer_holder {
    public:
        VarLayers model;

        vec_t feed_forward(const vec_t &a0);

        vec_t back_prop(const vec_t &error_, num_t lr);
    };

    void var_initialize_(layer& l){
        std::visit([&](auto &&arg) {
            arg.initialize();
        }, l);
    }

    template<typename ModelType>
    void var_initialize(ModelType &model) {
        for (auto &m: model) {
            var_initialize_(m);
        }
    }
    void var_set_training_(layer& l, bool training) {
        std::visit([&](auto &&arg) {
            arg.set_training(training);
        }, l);
    }

    template<typename ModelType>
    void var_set_training(ModelType &model, bool training) {
        for (auto &m: model) {
            var_set_training_(m, training);
        }
    }
    void var_update_weights_(layer &l, num_t train_percent) {
        std::visit([&](auto &&arg) {
            arg.update_weights(train_percent);
        }, l);
    }
    template<typename ModelType>
    void var_update_weights(ModelType &model, num_t train_percent) {
        for (auto &m: model) {
            var_update_weights_(m,train_percent);
        }
    }

    void var_start_batch_(layer &l) {
        std::visit([&](auto &&arg) {
            arg.start_batch();
        }, l);

    }
    template<typename ModelType>
    void var_start_batch(ModelType &model) {
        for (auto &m: model) {
            var_start_batch_(m);
        }
    }

    void var_end_batch_(layer &l) {
        std::visit([&](auto &&arg) {
            arg.end_batch();
        }, l);
    }
    template<typename ModelType>
    void var_end_batch(ModelType &model) {
        for (auto &m: model) {
            var_end_batch_(m);
        }
    }

    void var_start_sample_(layer &m) {
        std::visit([&](auto &&arg) {
            arg.start_sample();
        }, m);
    }

    template<typename ModelType>
    void var_start_sample(ModelType &model) {
        for (auto &m: model) {
            var_start_sample_(m);
        }
    }

    void var_end_sample_(layer &m) {
        std::visit([&](auto &&arg) {
            arg.end_sample();
        }, m);
    }
    template<typename ModelType>
    void var_end_sample(ModelType &model) {
        for (auto &m: model) {
            var_end_sample_(m);
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

    vec_t layer_holder::feed_forward(const vec_t &a0) {
        vec_t activation = a0;
        int at = 0;
        print_dbg("var input activations", activation.norm(), "@", 0);
        for (auto &l: model) {
            activation = var_forward(l, activation);
            print_dbg(at, var_get_name(l), "activation val", activation.norm(), activation.sum());
            ++at;
        }
        return activation;
    }

    vec_t layer_holder::back_prop(const vec_t &error_, num_t lr) {
        vec_t error = error_;
        int lix = model.size() - 1;
        print_dbg("err.", error.norm());
        for (auto cl = model.rbegin(); cl != model.rend(); ++cl) {
            error = var_layer_bp(*cl, error, lr);
            print_dbg(lix, var_get_name(*cl), "err val", error.norm(), error.sum());
            //assert(!has_nan(error));
            //assert(!has_inf(error));
            --lix;
        }
        return error;
    }

    VarLayers& get_iterable(VarLayers& model){
        return model;
    }
    layer& get_layer(layer& l){
        return l;
    }
}
#endif //NNNN_MODEL_H
