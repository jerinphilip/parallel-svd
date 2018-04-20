#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "storage.h"

//#define index(i, j, ld) (((j)*(ld)) + (i))
#define index(i, j, ld) (((j)*(ld)) + (i))

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
    
//    CPUTensor(): Tensor(0, 0) {};

    CPUTensor(int _rows, int _cols):
        Tensor(_rows, _cols) {
            storage = new CPUStorage(_size());
    }

    CPUTensor(const CPUTensor &B): Tensor(B.rows, B.cols){
        storage = new CPUStorage(_size());
        storage->_copy(B.storage);
    }

    bool is_zero(){
        double EPS = 1e-9;
        for(int i=0; i < rows; i++){
            for(int j=0; j <cols; j++){
                if (abs((*this)(i, j)) > EPS)
                    return false;
            }
        }
        return true;
    }

    void operator=(const CPUTensor A){
        /* 
         * TODO
         * Possible optimization, if dimensions equal, 
         * remove realloc overhead. 
         * */
        if(storage)
            delete storage;
        rows = A.rows;
        cols = A.cols;
        storage = new CPUStorage(_size());
        storage->_copy(A.storage);
    }

    CPUTensor flatten(){
        return reshape(rows*cols, 1);
    }

    CPUTensor reshape(int new_rows, int new_cols){
        assert( (rows * cols) == (new_rows * new_cols));

        /* Iterate through standard way. cols, then rows; */
        CPUTensor R(new_rows, new_cols);
        R.storage->_copy(storage);
//        memcpy(R.storage, storage, sizeof(double)*_size());
        return R;
    }
    
    /* transposes in place */
    void _transpose() {
        /* create new CPUStorage transposed */
        CPUStorage *transposed;
        transposed = new CPUStorage(_size());
    
        /* populate transposed */
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                transposed->data[index(i, j, rows)] = storage->data[index(i, j, rows)];
            }
        }
        
        /* replace storage with transposed */
        delete storage;
        storage = new CPUStorage(_size());
        storage->_copy(transposed);
        
        /* swap rows and cols for tensor */
        int tmp;
        tmp = rows;
        rows = cols;
        cols = tmp;
    }
    
    /* returns new transposed CPUTensor */
    CPUTensor transpose() {
        /* create new CPUStorage transposed */
        CPUStorage *transposed;
        transposed = new CPUStorage(_size());
    
        /* populate transposed */
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                transposed->data[index(j, i, cols)] = storage->data[index(i, j, rows)];
            }
        }
        
        /* create new CPUTensor, set its storage to transposed */
        CPUTensor T(cols, rows);
        T.storage->_copy(transposed);
        return T;
    }

    double& operator()(int i, int j){
        return storage->data[index(i, j, rows)];
    }

    double operator()(int i, int j) const {
        return storage->data[index(i, j, rows)];
    }

    ~CPUTensor(){
        delete storage;
    }


    static CPUTensor from_array(const double *A, int rows, int cols){
        CPUTensor C(rows, cols);
        C.storage->_dcopy(A, rows*cols);
        return C;
    }

    friend std::ostream& operator <<(std::ostream &out, const CPUTensor B){
        for(int i=0; i < B.rows; i++){
            for(int j=0; j < B.cols; j++){
                out << B(i, j) << " ";
            }
            out << "\n";
        }
        return out;
    }

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
