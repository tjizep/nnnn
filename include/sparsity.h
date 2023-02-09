//
// Created by Pretorius, Christiaan on 2022-10-01.
//

#ifndef NNNN_SPARSITY_H
#define NNNN_SPARSITY_H

#include <basics.h>
#include <vector>

#if defined(__aarch64__)
#include "sse2neon.h"
#else

#include <x86intrin.h>

#endif

namespace noodle {
    using namespace std;
    using namespace Eigen;


    struct block_sparsity {
        enum {
            block_size = 16 // only works with 4, 8, 16, 32, 64
        };

        struct block_entry {
            index_t row;
            index_t col;
            index_t size = block_size;

            block_entry(index_t row, index_t col) : row(row), col(col) {}

            block_entry(index_t row, index_t col, index_t size) : row(row), col(col), size(size) {}
        };

        struct value_block_entry {
            index_t row;
            index_t col;
            index_t size = block_size;

            std::array<num_t, block_size> data;

            void set_data(const num_t *d, index_t s) {
                //memset(&data[0], 0, block_size);
                memcpy(&data[0], d, s * sizeof(num_t));
                size = s;

            }

            value_block_entry(const block_entry &e) : row(e.row), col(e.col), size(e.size) {}

            value_block_entry(index_t row, index_t col) : row(row), col(col) {}

            value_block_entry(index_t row, index_t col, index_t size) : row(row), col(col), size(size) {}

            value_block_entry(index_t row, index_t col, const num_t *data, index_t size) : row(row), col(col) {
                set_data(data, size);
            }

            value_block_entry(index_t row, index_t col, const num_t *data) : row(row), col(col) {
                set_data(data, block_size);
            }
        };

        struct less_value_block_entry {
            inline bool operator()(const value_block_entry &x, const value_block_entry &y) {
                return std::tie(x.row, x.col) < std::tie(y.row, y.col);
            }
        };

        // the minimum actual sparsity required for block sparsity backprop optimizations to take place
        num_t opt_sparseness_threshold = 0.1;
        std::vector<block_entry> zero_blocks;
        std::vector<value_block_entry> valued_blocks;

        num_t sparseness = 0;
        num_t actual_sparseness = 0;

        num_t get_sparseness(const mat_t &w) const {
            return (num_t) (w.array() == 0).count() / (num_t) w.size();
        }

        num_t get_zeroes(const mat_t &w) const {
            return (num_t) (w.array() == 0).count();
        }

        num_t get_block_sparseness(const mat_t &weights) const {
            num_t zeroes = zero_blocks.size() * block_size;

            return zeroes / (num_t) weights.size();
        }

        void zero_weights(mat_t &weights) {
            index_t todo = 1;//zero_blocks.size()*0.05;

            for (const auto &zb: zero_blocks) {
                if (todo-- < 0)
                    weights.block<1, block_size>(zb.row, zb.col).array() = 0;

            }
        }

        void create_value_mask(const mat_t &weights) {
            valued_blocks.clear();
            index_t cols = weights.cols();
            index_t rows = weights.rows();
            for (index_t r = 0; r < rows; ++r) {
                for (index_t b = 0; b < cols; b += block_size) {

                    if (b + block_size < cols) {
                        auto block = weights.block<1, block_size>(r, b);
                        if ((block.array() == 0).count() < block_size) {
                            valued_blocks.push_back({r, b, &weights(r, b)});
                        }
                    } else {
                        index_t diff = b + block_size - cols;
                        assert(diff <= block_size);
                        valued_blocks.push_back({r, b, &weights(r, b), block_size - diff});
                    }
                }
            }
            std::sort(valued_blocks.begin(), valued_blocks.end(), less_value_block_entry());
        }

        void create_block_mask(const mat_t &weights) {
            zero_blocks.clear();
            valued_blocks.clear();
            index_t cols = weights.cols();
            index_t rows = weights.rows();
            for (index_t r = 0; r < rows; ++r) {
                for (index_t b = 0; b < cols; b += block_size) {

                    if (b + block_size < cols) {
                        auto block = weights.block<1, block_size>(r, b);
                        if ((block.array() == 0).count() == block_size) {
                            zero_blocks.push_back({r, b});
                        } else {
                            valued_blocks.push_back({r, b});
                        }
                    } else {
                        index_t diff = b + block_size - cols;
                        assert(diff <= block_size);
                        valued_blocks.push_back({r, b, block_size - diff});
                    }
                }
            }
            std::sort(valued_blocks.begin(), valued_blocks.end(), less_value_block_entry());
            actual_sparseness = get_block_sparseness(weights);
        }

        void copy_from_weights(const mat_t &weights) {
            for (auto &e: valued_blocks) {
                const num_t *pd = &weights(e.row, e.col);

                e.set_data(pd, e.size);
            }
        }

        __attribute__((noinline))
        void reduce_weights(mat_t &weights, index_t levl) {
            zero_weights(weights);
            copy_from_weights(weights);
            num_t prev_sparsity = get_block_sparseness(weights);
            if (prev_sparsity >= sparseness) {
                //cout << "3 sparsity " << get_sparseness(weights) << " " << get_block_sparseness(weights) << endl;
                return;
            }
            //cout << "prev_sparsity " << prev_sparsity << endl;
            if (valued_blocks.empty())
                create_value_mask(weights);
            const index_t total_blocks = weights.size() / block_size;
            num_t abs_mean = weights.array().abs().mean();
            num_t abs_max = weights.array().abs().maxCoeff();
            num_t abs_min = weights.array().abs().minCoeff();
            //if(levl == 1)
            //    cout << "min " << abs_min << " avg. " << abs_mean << " max " << abs_max << endl;s
            num_t high_water = abs_mean;
            num_t values = weights.size();
            num_t zeroed = 0;

            const num_t expand_factor = 4;
            for (index_t z = 0; z < 4; z++) {
                zeroed = 0;
                for (auto bl: valued_blocks) {
                    if (bl.size != block_size) {
                        continue;
                    }
                    auto block = weights.block<1, block_size>(bl.row, bl.col);
                    num_t b_avg = block.array().abs().mean();

                    if (b_avg < high_water) {//|| bl.row < weights.rows()/8
                        block.array() = 0;
                        zeroed++;
                        num_t current_sparseness = ((num_t) zeroed + zero_blocks.size()) * block_size / values;
                        // mTODO: discover when 15 should change
                        if (current_sparseness > sparseness || zeroed > total_blocks / 50.0) {
                            create_block_mask(weights);
                            zero_weights(weights);
                            copy_from_weights(weights);
                            return;
                        }
                    }
                }
                high_water *= expand_factor;
                if (high_water > abs_max) {
                    break;
                }
            } // for

            create_block_mask(weights);
            zero_weights(weights);
            copy_from_weights(weights);
            //cout << "2 sparsity " << get_sparseness(weights) << " " << get_block_sparseness(weights) << endl;
#if 0
            weights = weights.unaryExpr([&](num_t x) -> num_t{
                num_t x1 = abs(x) < highwater ? x*0.9999 : x;
                return abs(x1 ) < lowwater ? 0 : x1;
            });
#endif
        }

        __attribute__((noinline))
        void _base_project_mul_add(mat_t &o, const vec_t &l, const vec_t &r, num_t to_mul = 0) const {
            mat_t t = l * r.transpose();
            t *= to_mul;
            o += t;
        }

        inline static void vec_mad_f32(const int n, num_t *y, const num_t *x, const float v) {

            for (int i = 0; i < n; ++i) {
                y[i] += x[i] * v;
            }

        }

        inline static num_t vec_sum_mad_f32(const int n, const num_t *y, const float v) {
            num_t r = 0.0;
            for (int i = 0; i < n; ++i) {
                r += y[i] * v;
            }
            return r;
        }

        template<int N>
        inline static void vec_mad_f32_n(num_t *a, const num_t *b, const float v) {

#ifdef __AVX__
            __m256 num_b, num_a, mmul, scalar;
            scalar = _mm256_set1_ps(v);  // broadcasts scalar to v
            for (int i = 0;
                 i < N; i += 8) { // I think this gives the compiler a hint to unroll the loop since its a constant
                num_b = _mm256_loadu_ps(b + i);
                num_a = _mm256_loadu_ps(a + i);
                mmul = _mm256_mul_ps(num_b, scalar);
                _mm256_storeu_ps(a + i, _mm256_add_ps(num_a, mmul));
            }
#elif __SSE2__ || defined(__aarch64__) //using sse2 -> neon
            __m128 num_b, num_a, mmul, scalar;
            scalar = _mm_set1_ps(v);  // broadcasts scalar to v
            for (auto i = 0; i < N; i += 4) { // I think this gives the compiler a hint to unroll the loop since its a constant
                num_b = _mm_loadu_ps(b + i);
                num_a = _mm_loadu_ps(a + i);
                mmul = _mm_mul_ps(num_b, scalar);
                _mm_store_ps(a + i, _mm_add_ps(num_a, mmul));
            }
#else
            for (int i = 0; i < N; ++i) {
                y[i] += x[i]*v;
            }
#endif
        }

        mutable std::vector<num_t> temp_r; // << makes a big difference in perf by not malloc'ing repeatedly (not something the optimizer will do by itself)
        /**
         * main component of fc backpropagation
         * @param o
         * @param l
         * @param r
         * @param to_mul
         */
        __attribute__((noinline))
        void project_mul_add(mat_t &o, const vec_t &l, const vec_t &r, num_t to_mul = 0) {
            if (o.size() < l.size()) {
                o = mat_t::Zero(l.rows(), r.rows());
            }

            if (!valued_blocks.empty() && (block_size % 8) == 0 && actual_sparseness > opt_sparseness_threshold) {

                array<num_t, block_size> old;
                const num_t *pl = &l(0);
                temp_r.resize(r.rows() + 16);
                num_t *temp_a = &temp_r[0];
                for (index_t i = 0; i < temp_r.size(); ++i) {
                    temp_a[i] = to_mul * r(i);
                }

                for (auto &e: valued_blocks) {

                    num_t rval = pl[e.row];
                    num_t *pd = &o(e.row, e.col);

                    if (e.size == block_size) {
                        vec_mad_f32_n<block_size>(pd, &temp_a[e.col], rval);
                    } else
                        vec_mad_f32(e.size, pd, &temp_a[e.col], rval);
                }
            } else {
                _base_project_mul_add(o, l, r, to_mul);
            }
        }

        __attribute__((noinline))
        void _base_vec_mul_assign(vec_t &o, const mat_t &l, const vec_t &r) {
            o = l * r;
        }

        inline static num_t vec_dot_f32(const int n, const num_t *x, const num_t *y) {
            num_t sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += (x[i] * y[i]);
            }
            return sum;
        }


        template<int N>
        inline static num_t vec_dot_f32_n(const num_t *a, const num_t *b) {

#ifdef __AVX__
            __m256 n1, n2, n3, sum;
            sum = _mm256_setzero_ps();  //sets sum to zero
            for (int i = 0; i < N; i += 8) {
                n1 = _mm256_loadu_ps(a + i);   //loads unaligned array a into num1  num1= a[3]  a[2]  a[1]  a[0]
                n2 = _mm256_loadu_ps(b + i);   //loads unaligned array b into num2  num2= b[3]   b[2]   b[1]  b[0]
                n3 = _mm256_dp_ps(n1, n2, 0xFF);
                sum = _mm256_add_ps(sum, n3); // vertical sum
            }
            return (num_t) sum[0] + sum[4];
#elif __SSE2__ || defined(__aarch64__) //using sse2 -> neon
            float total;
            int i;
            __m128 n1, n2, n3, n4;
            n4 = _mm_setzero_ps();  //sets sum to zero
            for (i = 0; i < N; i += 4) {
                n1 = _mm_loadu_ps(a + i);   //loads unaligned array a into num1  num1= a[3]  a[2]  a[1]  a[0]
                n2 = _mm_loadu_ps(b + i);   //loads unaligned array b into num2  num2= b[3]   b[2]   b[1]  b[0]
                n3 = _mm_mul_ps(n1, n2); //performs multiplication   num3 = a[3]*b[3]  a[2]*b[2]  a[1]*b[1]  a[0]*b[0]
                n3 = _mm_hadd_ps(n3, n3); //performs horizontal addition
                //num3=  a[3]*b[3]+ a[2]*b[2]  a[1]*b[1]+a[0]*b[0]  a[3]*b[3]+ a[2]*b[2]  a[1]*b[1]+a[0]*b[0]
                n4 = _mm_add_ps(n4, n3);  //performs vertical addition
            }
            n4 = _mm_hadd_ps(n4, n4);
            _mm_store_ss(&total, n4);
            return total;
#else
            float total = 0.0;
            for (int i = 0; i < N; ++i) {
                total += a[i] * b[i];
            }
            return (num_t)total;
#endif
        }


        /**
         * used in forward propagation
         * @param o
         * @param l
         * @param r
         */
        __attribute__((noinline))
        void vec_mul_assign(vec_t &o, const mat_t &l, const vec_t &r) {

            if (!valued_blocks.empty() && (block_size % 8) == 0 && actual_sparseness > 0.3) {
                o.resize(l.rows(), 1);
                o.setZero(); /// because its assign not update
                const num_t *pr = &r(0, 0);

                index_t r_size = r.rows();
                num_t *po = &o(0, 0);
                num_t dot = 0.0;

                index_t currow = valued_blocks.begin()->row;

                for (auto &e: valued_blocks) {
                    if (currow != e.row) {
                        po[currow] = dot;
                        currow = e.row;
                        dot = 0;
                    }

                    if (e.size == block_size) {
                        dot += vec_dot_f32_n<block_size>(e.data.data(), pr + e.col);
                    } else {
                        dot += vec_dot_f32(e.size, e.data.data(), pr + e.col);
                    }
                }
                po[currow] = dot;
            } else {

                _base_vec_mul_assign(o, l, r);
            }
        }

        /**
         * sparse optimized value masked matrix transpose multiply
         * only multiplies blocks
         * @param result
         * @param weights
         * @param error
         */
        __attribute__((noinline))
        void mask_mul(vec_t &result, const mat_t &weights, const vec_t &error) const {
            // if weights == 50x100
            //  100x1 = 100x50 * 50x1
            // result = weights.transpose() * error
            if (!valued_blocks.empty() && (block_size % 16) == 0 && actual_sparseness > opt_sparseness_threshold) {
                result = vec_t::Zero(weights.cols(), 1);
                for (auto e: valued_blocks) {
                    num_t mr = error(e.row, 0);
                    num_t * pr = &result(e.col, 0);
                    const num_t * pl = &weights(e.row, e.col);
                    if(e.size == block_size){
                        vec_mad_f32_n<block_size>(pr, pl, mr);// << auto loop unrolled version
                    }else
                        vec_mad_f32(e.size, pr, pl, mr);
                }
          } else {
                result = weights.transpose() * error;
            }
        }
    };
}
#endif //NNNN_SPARSITY_H
