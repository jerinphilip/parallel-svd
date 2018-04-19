#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>

struct Tensor {
    int rows, cols;
    Tensor(int _rows, int _cols): 
        rows(_rows), cols(_cols){
    }

    int _size(){
        return rows*cols;
    }
};

struct CPUTensor: public Tensor {
    CPUStorage *storage;

    CPUTensor(int _rows, int _cols):
        Tensor(_rows, _cols) {
            storage = new CPUStorage(_size());
    }

    CPUTensor(const CPUTensor &B): Tensor(B.rows, B.cols){
        storage = new CPUStorage(_size());
        storage->_copy(B.storage);
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
                transposed->data[j*rows + i] = storage->data[i*cols + j];
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
                transposed->data[j*rows + i] = storage->data[i*cols + j];
            }
        }
        
        /* create new CPUTensor, set it's storage to transposed */
        CPUTensor T(cols, rows);
        T.storage->_copy(transposed);   
        return T;
    }    

    double& operator()(int i, int j){
        return storage->data[i*cols + j];
    }

    double operator()(int i, int j) const {
        return storage->data[i*cols + j];
    }

    ~CPUTensor(){
        delete storage;
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

#endif
