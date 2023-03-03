//
// Created by kriso on 2/25/2023.
//

#ifndef NNNN_GRAPH_H
#define NNNN_GRAPH_H
#include <model.h>
#include <string>

namespace noodle{
    using namespace std;
    using namespace Eigen;
    // graph node
    struct node {
        string name;
        vector<string> source;
        vector<index_t> destinations;
        vector<index_t> sources;

        /// results for forward operation
        vec_t activation = row_vector();
        vec_t output = row_vector();
        vector<vec_t> errors;

        /// result for gradient backpropagation
        vec_t bp_input = row_vector(); // aka gradient
        vec_t bp_output = row_vector();

        index_t outputs{0};
        index_t inputs{0};
        layer operation{empty_layer{}};
        index_t index{-1};

        operator layer&(){
            return operation;
        }
        operator const layer&() const {
            return operation;
        }

        void clear() {
            *this = node();
        }

        node() {

        }

        bool contains_source(index_t s) const {
            for (index_t i: sources) {
                if (i == s) return true;
            }
            return false;
        }

        bool contains_destination(index_t d) const {
            for (index_t i: destinations) {
                if (d == i) return true;
            }
            return false;
        }

        bool empty() const {
            return index < 0;
        }
    };

    typedef unordered_map<string, index_t> NameIndex;
    typedef vector<node> Nodes;
    typedef unordered_set <string> NameSet;

    /// a 'safer' 'limited physical pointer' differential graph
    /// this class would be safer if we could limit node
    /// pointers outside of this class
    struct graph {
        enum {
            null_node = -1
        };
        node empty;
        Nodes nodes;
        NameIndex index;
        void update_layers(graph &dest) const  {
            auto isource = nodes.begin();
            auto idest = dest.nodes.begin();
            for (; isource != nodes.end() && idest != dest.nodes.end(); ++isource, ++idest) {
                if (!var_layer_update_bp(idest->operation, isource->operation)) {
                    print_err("layer type not found");
                }
            }
        }
        void start_sample(){
            for (auto &m: nodes) {
                std::visit([&](auto &&arg) {
                    arg.start_sample();
                }, m.operation);
            }
        }
        void end_sample(){
            for (auto &m: nodes) {
                std::visit([&](auto &&arg) {
                    arg.end_sample();
                }, m.operation);
            }
        }
        bool build_sources() {
            for (auto &n: nodes) {
                if (n.source.size() != n.sources.size()) {
                    print_dbg(n.name, n.index);
                    n.sources.clear();
                    for (const auto &s: n.source) {

                        index_t ix = id(s);
                        if (ix < 0) {
                            print_err("could not find id", ix, "for", s);
                            return false;
                        }
                        print_dbg(n.name, n.index, ":", s, ix);
                        n.sources.push_back(ix);
                    }
                }
            }
            return true;
        }
        void clear_activations() {
            for (auto &n: nodes) {
                //n.activations.clear();
            }
        }
        bool build_destinations() {
            if (!build_sources()) {
                return false;
            }
            for (auto &n: nodes) {
                n.destinations.clear();
            }
            for (auto &n: nodes) {
                print_dbg(n.name, n.index, n.source.size());
                for (const string &s: n.source) {
                    print_dbg(n.name, ":", s);
                    index_t ix = id(s);
                    if (resolve(ix).empty()) {
                        print_err("invalid node name", s);
                        return false;
                    }

                    if (n.contains_destination(n.index)) {
                        print_err("duplicate destination", n.name, n.index);
                        return false;
                    }
                    resolve(ix).destinations.push_back(n.index);
                }
            }
            return true;
        }


        void clear() {
            empty.clear();
            nodes.clear();
            index.clear();
        }

        index_t id(const string &name) {
            return resolve(name).index;
        }

        index_t add(const node &n) {
            auto i = index.find(n.name);
            index_t x;
            if (i == index.end()) {
                x = nodes.size();
                nodes.push_back(n);
                index[n.name] = x;
            } else {
                x = i->second;
                nodes[x] = n;
            }
            nodes[x].index = x;
            return x;
        }

        node &resolve(index_t ix) {
            if (ix >= 0 && ix < nodes.size()) return nodes[ix];
            return empty;
        }

        const node &resolve(index_t ix) const {
            if (ix >= 0 && ix < nodes.size()) return nodes[ix];
            return empty;
        }

        node &resolve(const string &name) {
            auto i = index.find(name);
            if (i != index.end()) {
                return resolve(i->second);
            }
            return empty;
        }

        const node &resolve(const string &name) const {
            auto i = index.find(name);
            if (i != index.end()) {
                return resolve(i->second);
            }
            return empty;
        }

        bool is_empty(const node &n) {
            return n.empty() || &n == &empty;
        }

        bool validate_sources() const {
            for (auto &n: nodes) {
                for (auto d: n.destinations) {
                    auto ds = std::find(resolve(d).sources.begin(), resolve(d).sources.end(), n.index);
                    if (ds == resolve(d).sources.end()) {
                        print_err("source", n.index, "not found in node", d);
                        return false;
                    }
                }
            }
        }

        NameIndex::const_iterator validate_source(index_t l) const {
            if (resolve(l).source.empty()) {
                return index.end();
            }
            for (auto source: resolve(l).source) {

                auto i = index.find(source);
                if (i == index.end()) {
                    print_err("source not found", source);
                    return index.end();
                }
                return i;
            }
            return index.end();
        }

        uint32_t find_outputs(const std::string name) const {
            return find_outputs(index.find(name));
        }

        uint32_t find_outputs(NameIndex::const_iterator is, NameSet names = {}) const {
            if (is == index.end()) {
                print_err("source not found");
                return 0;
            }
            const auto l = is->second;
            if (resolve(l).empty()) {
                print_err("source not found");
                return 0;
            }
            if (names.contains(resolve(l).name)) {
                print_err("name cycle detected at", resolve(l).name);
                return 0;
            }
            if (resolve(l).outputs == 0) {
                auto i = validate_source(l);
                return find_outputs(i, names);
            }
            print_inf("outputs", resolve(l).outputs, "for", resolve(l).name);
            return resolve(l).outputs;
        }
        struct base_selector{
            std::vector<index_t> pointers;
            index_t get() const {
                index_t r = (pointers.empty()) ? null_node : *(pointers.begin());
                print_dbg("get", r);
                return r;
            }
            node& resolve(graph& g){
                return g.resolve(get());
            }
            const node& resolve(const graph& g) const {
                return g.resolve(get());
            }
            bool ok(const graph &g) const {
                return !(g.resolve(get()).empty());
            }

        };
        // breadth first forward iterator
        struct forward_selector : public base_selector {
            vector<index_t>& destinations(){
                return pointers;
            }
            const vector<index_t>& destinations() const {
                return pointers;
            }
            forward_selector(){

            }
            forward_selector(index_t initial) {
                destinations().push_back(initial);
            }
            forward_selector(vector<index_t> destinations) {
                this->destinations() = destinations;
            }

            void set_activation(graph& model, const vec_t &activation){
                if(!resolve(model).sources.empty()){
                    print_err("this is not a root node");
                    return;
                }
                resolve(model).activation = activation;
            }
            vec_t& get_activation(graph& model){
                for(auto s : resolve(model).sources){
                    return model.resolve(s).output;
                }
                return resolve(model).activation;
            }
            vector<vec_t> activations;
            bool get_activations(graph& model){
                activations.clear();
                if(resolve(model).sources.size() < 2){
                    print_err("requires multiple activations");
                    return false;
                }
                for(auto s : resolve(model).sources){
                    activations.push_back(model.resolve(s).output);
                }
                return !activations.empty();
            }

            bool forward(graph& model){
                index_t inputs = resolve(model).inputs;
                index_t outputs = resolve(model).outputs;
                //print_dbg(inputs,outputs,get_activation(model).rows());
                if(inputs > 0 && get_activation(model).rows() > 0 && inputs != get_activation(model).rows()){
                    print_err("the required vector input size (inputs)",resolve(model).inputs,"does not match the given",get_activation(model).rows());
                    return false;
                }
                if(resolve(model).sources.size() > 1){
                    get_activations(model);
                    resolve(model).output = var_forward(resolve(model).operation, activations);
                }else{
                    resolve(model).output = var_forward(resolve(model).operation, get_activation(model));
                }
#if 0
                if(outputs != resolve(model).output.rows()){
                    print_err("output size (outputs)",outputs,"does not match the given",resolve(model).output.rows());
                    return false;
                }
#endif
                return true;
            }

            vec_t vforward(graph& model, vec_t& activation){
                vec_t r = var_forward(resolve(model).operation, activation);
                for(auto d : resolve(model).destinations){
                    //model.resolve(d).activations.push_back(activation);
                }
                return r;

            }

            bool next(const graph &g) {
                index_t current = get();
                print_dbg("current", current);
                destinations().erase(destinations().begin());
                for (auto i: g.resolve(current).destinations) {
                    destinations().push_back(i);
                }
                return true;
            }
        };

        struct reverse_selector : public base_selector {
            vector<vec_t> errors; /// when we are collecting multiple sources
            vector<index_t>& sources(){
                return pointers;
            }
            const vector<index_t>& sources() const {
                return pointers;
            }
            reverse_selector(){
            }
            reverse_selector(index_t initial) {
                sources().push_back(initial);
            }
            reverse_selector(vector<index_t> sources) {
                this->sources() = sources;
            }

            void set_error(graph& model, const vec_t & error){
                if(resolve(model).sources.empty()){
                    print_err("this is not a end node");
                    return;
                }
                resolve(model).bp_input = error;
            }
            vec_t& get_error(graph& model){
                for(auto s : resolve(model).destinations){
                    return model.resolve(s).bp_output;
                }
                return resolve(model).bp_input;
            }

            vector<vec_t>& get_errors(graph& model){
                errors.clear();
                if(resolve(model).destinations.size() < 2){
                    print_err("requires multiple activations");
                    return errors;
                }
                for(auto s : resolve(model).destinations){
                    errors.push_back(model.resolve(s).bp_output);
                }
                return errors;
            }

            bool next(const graph &g) {
                if (sources().empty()) return false;
                index_t current = get();
                sources().erase(sources().begin());
                for (auto i: g.resolve(current).sources) {
                    sources().push_back(i);
                }
                return true;
            }
            /// note: the role names "destinations" and "sources" are used in the forward
            /// sense so in the backward sense their respective roles reverses
            bool backward(graph& model, num_t learning_rate){
                index_t inputs = resolve(model).inputs;
                index_t outputs = resolve(model).outputs;
                //print_dbg(inputs,outputs,get_activation(model).rows());
                if(outputs > 0 && get_error(model).rows() > 0 && outputs != get_error(model).rows()){
                    print_err("the required output vector size (outputs)",outputs,"does not match the given",get_error(model).rows());
                    return false;
                }
                /// "destinations" are realy the sources of the backprop operation
                if(resolve(model).destinations.size() > 1){
                    resolve(model).bp_output = var_layer_bp(resolve(model).operation, get_errors(model), learning_rate);
                }else{
                    resolve(model).bp_output = var_layer_bp(resolve(model).operation, get_error(model),learning_rate);
                }
#if 0
                if(outputs != resolve(model).output.rows()){
                    print_err("output size (outputs)",outputs,"does not match the given",resolve(model).output.rows());
                    return false;
                }
#endif
                return true;
            }
        };

        forward_selector first() const {
            vector<index_t> destinations;
            for (auto &n: nodes) {
                if (n.source.empty()) {
                    destinations.push_back(n.index);
                }
            }
            return {destinations};
        }

        reverse_selector last() const {
            vector<index_t> sources;
            for (auto &n: nodes) {
                if (n.destinations.empty()) {
                    sources.push_back(n.index);
                }
            }
            return {sources};
        }
        typedef Nodes::iterator iterator;
        iterator begin() {
            return nodes.begin();
        }
        iterator end(){
            return nodes.end();
        }

        void initialize_operators(){
            for (auto &m: nodes) {
                var_initialize_(m.operation);
            }
        }
    };
    Nodes & get_iterable(graph& model){
        return model.nodes;
    }

    layer& get_layer(node& l){
        return l.operation;
    }
}
#endif //NNNN_GRAPH_H
