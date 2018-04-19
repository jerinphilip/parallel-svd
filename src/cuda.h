#ifndef CUDA_H
#define CUDA_H

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>
#include <cassert>


class CUDAContext {
    private:
        cublasHandle_t _handle;
        static CUDAContext* instance;
        CUDAContext();

    public:
        static CUDAContext* getInstance();
        CUDAContext(const CUDAContext&) = delete;
        CUDAContext& operator=(const CUDAContext&) = delete;
        cublasHandle_t handle(){ return _handle; }
};


CUDAContext* CUDAContext::instance = 0;

CUDAContext::CUDAContext(){
    float *devPtrA;
    bool alloc, handle_creation;
    alloc = cudaMalloc((void**)&devPtrA, sizeof(float));
    handle_creation = cublasCreate(&_handle);
    assert (alloc == cudaSuccess);
    assert (handle_creation == CUBLAS_STATUS_SUCCESS);

}

CUDAContext* CUDAContext::getInstance(){
    if(instance == 0){
        instance = new CUDAContext();
    }
    return instance;

}

#endif
