#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <Eigen/Dense>

namespace mnist {


    using namespace std;
    using namespace Eigen;


    static uint32_t swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }

    static std::vector<int> get_labels(string path) {
        ifstream file(path, ios::in | ios::binary);
        uint32_t magic;
        uint32_t num_items;
        char label;
        vector<int> labels;
        if (file.is_open()) {
            print_inf("Reading label file:", path);

            file.read(reinterpret_cast<char *>(&magic), 4);
            magic = swap_endian(magic);
            if (magic != 2049) {
                fatal_err("unexpected magic number:", magic, "(expecting 2049)");
                return labels;
            }

            file.read(reinterpret_cast<char *>(&num_items), 4);
            num_items = swap_endian(num_items);

            for (uint32_t item_id = 0; item_id < num_items; item_id++) {
                file.read(&label, 1);
                labels.push_back(label);
            }
        } else {
            fatal_err("could not read label file", path);
        }
        return labels;
    }

    template<typename vec_t>
    static std::vector<vec_t> get_images(string path, uint32_t vsize) {
        ifstream file(path, ios::in | ios::binary);
        uint32_t magic;
        uint32_t image_count;
        uint32_t rows;
        uint32_t cols;
        vector<uint8_t> pixels;
        vector<vec_t> images;
        unordered_map<uint32_t, uint32_t> log_hist;
        if (file.is_open()) {
            print_inf("Reading image file:", path);

            file.read(reinterpret_cast<char *>(&magic), 4);
            magic = swap_endian(magic);
            if (magic != 2051) {
                fatal_err("surprising magic number:", magic, "(expexted 2051)");
                return images;
            }

            file.read(reinterpret_cast<char *>(&image_count), 4);
            image_count = swap_endian(image_count);
            file.read(reinterpret_cast<char *>(&rows), 4);
            rows = swap_endian(rows);
            file.read(reinterpret_cast<char *>(&cols), 4);
            cols = swap_endian(cols);

            print_inf("Image Count:", image_count);
            print_inf("Image Size:", rows, "x", cols);

            pixels.resize(rows * cols);
            size_t at = 0;
            size_t max_at = 0;
            const float total_pixels = rows * cols;
            const float VECT_SIZE = vsize;

            for (uint32_t i = 0; i < image_count; i++) {
                file.read((char *) pixels.data(), rows * cols);
                vec_t image = vec_t::Constant(VECT_SIZE, 0);
                size_t at = 0;
                uint32_t offset = 0;
                //while(pixels[offset] == 0) ++offset;
                //offset -= (offset % cols);

                for (uint32_t j = offset; j < total_pixels; ++j) {
                    float pj = pixels[j];
                    //float pj = exp(floor(log(pixels[j])));;
                    //if(j % cols == 0) cout << endl;
                    //cout.width(4);
                    //cout << pj;
                    uint32_t ip = floor((j - offset) * VECT_SIZE / total_pixels);
                    //uint32_t lp

                    //log_hist[lp]++;
                    image[ip] += pj / total_pixels;
                }
                //cout << endl;
#if 0
                uint32_t vcols = VECT_SIZE * num_cols / total_pixels;
                for (uint32_t j = 0; j < VECT_SIZE; ++j) {
                    if(j % vcols == 0) cout << endl;
                    cout.width(5);
                    cout << floor(image[j]*total_pixels/4);
                }
                cout << endl;

#endif
                max_at = std::max<size_t>(at, max_at);
                images.push_back(image);
            }
            for (auto p: log_hist) {
                print_inf(p.first, p.second);
            }
            print_inf(max_at);
        } else {
            fatal_err("Failure reading image file:", path);
        }
        return images;
    }

    template<typename vec_t>
    static vec_t output_vector(int label) {
        vec_t y = vec_t::Constant(10, 0);
        y[label] = 1;
        return y;
    }

    template<typename vec_t>
    static vector<vec_t> get_output_vectors(vector<int> labels) {
        vector<vec_t> output_vectors;

        for (uint32_t i = 0; i < labels.size(); i++) {
            output_vectors.push_back(output_vector<vec_t>(labels[i]));
        }

        return output_vectors;
    }
};