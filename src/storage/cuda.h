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


#endif
