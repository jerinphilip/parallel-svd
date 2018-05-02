#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "storage/storage.h"

#define index(i, j, ld) (((j)*(ld)) + (i))

struct Tensor;
struct CPUTensor;
struct CUDATensor;

struct Tensor {
    int rows, cols;
    Tensor(int _rows, int _cols): 
        rows(_rows), cols(_cols){
    }

    int _size() const{
        return rows*cols;
    }
};

struct CPUTensor: public Tensor {
    CPUStorage *storage;
    CPUTensor(int _rows, int _cols);
    CPUTensor(const CPUTensor &B);
    CPUTensor();
    bool is_zero();
    bool is_diagonal();
    void operator=(const CPUTensor A);
    CPUTensor flatten();
    CPUTensor reshape(int new_rows, int new_cols);
    void _transpose();
    CPUTensor transpose();
    CUDATensor gpu() const;
    double& operator()(int i, int j);
    double operator()(int i, int j) const;
    ~CPUTensor();
    static CPUTensor from_array(const double *A, int rows, int cols);
    friend std::ostream& operator <<(std::ostream &out, const CPUTensor B);
    
};

struct CUDATensor : public Tensor {
    CUDAStorage *storage;
    CUDATensor(int _rows, int _cols);
    CUDATensor(CPUTensor C);
    CUDATensor();
    CUDATensor(const CUDATensor &B);
    void _copy(CUDATensor *B);
    CPUTensor cpu() const;
    void operator=(const CUDATensor &B);
    CUDATensor transpose();
    CUDATensor flatten();
    CUDATensor reshape(int, int);
    friend std::ostream& operator <<(std::ostream &out, const CUDATensor B);
    static CUDATensor from_array(double *A, int rows, int cols);
    ~CUDATensor();
    bool is_diagonal();
};

#endif
