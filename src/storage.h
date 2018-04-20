#ifndef STORAGE_H
#define STORAGE_H

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "cuda.h"

struct Storage {
    double *data;
    int size;
    Storage(int size): size(size){}
};

struct CPUStorage : public Storage {
    CPUStorage(int size): Storage(size){
        data = (double*)(std::malloc)(sizeof(double)*size);
        memset(data, 0, size);
    }

    void _copy(CPUStorage *b){
        assert (size == b->size);
        memcpy(data, b->data, sizeof(double)*size);
    }

    void _dcopy(const double *d, int _size){
        assert (size == _size);
        memcpy(data, d, sizeof(double)*size);
    }

    ~CPUStorage(){
        std::free(data);
    }

};

struct CUDAStorage: public Storage {
    static CUDAContext *ctx;
    CUDAStorage(int size): Storage(size){
        int status;
        status = cudaMalloc((void**)&data, sizeof(double)*size);
        /* TODO assertions on status */
        assert(status == cudaSuccess);

    }

    void _copy(CUDAStorage *b){
        int incx=1, incy=1, status;
        status = cublasDcopy(ctx->handle(), size, b->data, incx, data, incy);
        assert (status == CUBLAS_STATUS_SUCCESS);
    }

    ~CUDAStorage(){
        cudaFree(data);
    }

};

CUDAContext* CUDAStorage::ctx = CUDAContext::getInstance();

#endif
