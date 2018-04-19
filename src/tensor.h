#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>


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

    CPUTensor flatten(){
        return reshape(rows*cols, 1);
    }

    CPUTensor reshape(int new_rows, int new_cols){
        assert( (rows * cols) == (new_rows * new_cols));

        /* Iterate through standard way. cols, then rows; */
        CPUTensor R(new_rows, new_cols);
        memcpy(R.storage, storage, sizeof(double)*_size());
        return R;
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
