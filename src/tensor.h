#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>

struct range {
    int start, end;
    range(): start(-1), end(-1){}
    range(int _start, int _end): start(_start), end(_end) {}
    bool isset(){ return start != -1; }
    int size() const { assert(end !=-1); return end - start; }
};

struct block {
    range row, col;
    block operator()(range r){
        assert (not (row.isset() and col.isset()));

        if (not row.isset()) { row = r; }
        else { col = r; }
        return *this;
    }

    block operator()(int x){
        return (*this)(x, x+1);
    }

    block operator()(int x, int y){
        assert (not (row.isset() and col.isset()));
        range r;
        r.start = x, r.end = y;
        return (*this)(r);
    }


};

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
