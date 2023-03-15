#pragma once
//#define EIGEN_DONT_VECTORIZE
#include <logging/console.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include <fstream>
#include <string.h>
#include <fstream>
#include <random>
#include <cassert>


namespace noodle {
    using namespace std;
    using namespace Eigen;

    typedef float num_t;
    typedef int64_t index_t;
    typedef VectorXf vec_t;

    typedef Matrix<num_t, Dynamic, Dynamic, RowMajor> mat_t;//MatrixXf
    typedef SparseMatrix<num_t> sparse_mat_t;

    static inline vec_t row_vector() { return vec_t::Constant(0, 0); }

    static inline mat_t matrix() { return mat_t::Constant(0, 0, 0); }

    static inline sparse_mat_t NULL_SM() { return sparse_mat_t(0, 0); }

    template<typename vT>
    static inline bool has_nan(vT &v) {
        return !(((v.array() == v.array())).all());
    }

    template<typename vT>
    static inline bool has_inf(vT &v) {
        return !(((v - v).array() == (v - v).array()).all());
    }

    // correct assign addition
    template<typename bT>
    static inline void assign_add(bT &a, const bT &b) {
        if (a.size() == 0) a = b;
        else a += b;
    }
    template<typename bT>
    static inline void round(bT& a, index_t places){
        num_t fact = pow(10, places);
        a.array() *= fact;
        a.array() = a.array().round()/fact;
    }

    static mat_t constant(const mat_t &original, num_t value) {
        return mat_t::Constant(original.rows(), original.cols(), value);
    }

    static mat_t m_pow(const mat_t &original, num_t value) {
        return original.array().pow(value).matrix();
    }

    static mat_t m_sqrt(const mat_t &original) {
        return original.cwiseSqrt();
    }

    inline string qt(const string& value){
        return "'"+value+"'";
    }
};