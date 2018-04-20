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
    bool is_zero();
    void operator=(const CPUTensor A);
    CPUTensor flatten();
    CPUTensor reshape(int new_rows, int new_cols);
    void _transpose();
    CPUTensor transpose();
    double& operator()(int i, int j);
    double operator()(int i, int j) const;
    ~CPUTensor();
    static CPUTensor from_array(const double *A, int rows, int cols);
    friend std::ostream& operator <<(std::ostream &out, const CPUTensor B);
    
};

struct CUDATensor : public Tensor {
    CUDAStorage *storage;
    CUDATensor(int _rows, int _cols): Tensor(_rows, _cols){
        storage = new CUDAStorage(_size());
    }

    void _copy(CUDATensor *B){
        storage->_copy(B->storage);
    }

    CUDATensor(CPUTensor C): Tensor(C.rows, C.cols){
        storage = new CUDAStorage(_size());
        int status;
        status = cublasSetMatrix(rows, cols, sizeof(double), 
                C.storage->data,
                rows, 
                storage->data, rows);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    CPUTensor cpu() const{
        double *buffer;
        buffer = NULL;
        buffer = (double*)std::malloc(_size()*sizeof(double));
        int status;
        status = cublasGetMatrix(rows, cols, 
                sizeof(double), storage->data, 
                rows, 
                buffer, rows);
        assert(status == CUBLAS_STATUS_SUCCESS);
        CPUTensor C = CPUTensor::from_array(buffer, rows, cols);
        std::free(buffer);
        return C;
    }

    friend std::ostream& operator <<(std::ostream &out, const CUDATensor B){
        out << B.cpu();
        return out;
    }
};

#endif
