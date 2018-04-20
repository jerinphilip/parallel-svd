#include "cuda.h"

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
