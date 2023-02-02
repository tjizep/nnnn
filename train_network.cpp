
#include <network.h>
#include <read_mnist.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {
    using namespace std;
    string data_dir = "../data/";
#ifdef EIGEN_VECTORIZE
    cout << "'Eigen' Vectorization is enabled" << endl;
#else
    cout << "'Eigen' Vectorization is disabled" << endl;
#endif
    size_t num_epochs = 24 ;
    noodle::num_t momentum = 0;
    noodle::num_t leakiness = 40;
    uint32_t model_size = 100;
    noodle::VarLayers model;
    noodle::num_t sparsity = 0.6;


    model.push_back(noodle::fc_layer{180, model_size, sparsity, momentum});
    model.push_back(noodle::relu_layer{leakiness});

    model.push_back(noodle::fc_layer{model_size, model_size/2, sparsity, momentum});
    model.push_back(noodle::relu_layer{leakiness});
    //model.push_back(noodle::pepper_layer{0.2});

    model.push_back(noodle::fc_layer{model_size/2, model_size/2,sparsity, momentum});
    model.push_back(noodle::relu_layer{leakiness});
    //model.push_back(noodle::low_sigmoid_layer{0.36});// TODO: make this a learned parameter
    //model.push_back(noodle::dropout_layer{0.2});
    model.push_back(noodle::fc_layer{model_size/2, 10, sparsity/1.2f, momentum});
    model.push_back(noodle::soft_max_layer{});

    //model.push_back(noodle::low_sigmoid_layer{});

    vector<int> training_labels = mnist::get_labels(data_dir+"train-labels-idx1-ubyte");
    vector<noodle::vec_t> training_inputs = mnist::get_images<noodle::vec_t>(data_dir+"train-images-idx3-ubyte");
    vector<noodle::vec_t> training_outputs = mnist::get_output_vectors<noodle::vec_t>(training_labels);

    vector<int> test_labels = mnist::get_labels(data_dir+"t10k-labels-idx1-ubyte");
    vector<noodle::vec_t> test_inputs = mnist::get_images<noodle::vec_t>(data_dir+"t10k-images-idx3-ubyte");

    size_t mini_batch_size = 24;
    array<noodle::num_t,2>  learning_rate = {0.01, 0.00002};

    cout << "batch in_size: " << mini_batch_size << " learning rate: " << learning_rate[0] << " #epochs: " << num_epochs << endl;
    noodle::trainer n(training_inputs, training_outputs, test_inputs, test_labels, mini_batch_size,
                      learning_rate);

    n.stochastic_gradient_descent(num_epochs, model, 4, 75);
    int ok = n.save_weights_and_biases("weights_and_biases.txt");
    if (ok == 0) {
        cout << "Save OK." << endl;
    } else {
        cout << "Save Fail." << endl;
    }
    return 0;
}
