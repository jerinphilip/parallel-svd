#include "storage.h"

CUDAStorage::CUDAStorage(int size): Storage(size){
    int alloc, mset;
    alloc = cudaMalloc((void**)&data, sizeof(double)*size);
    /* TODO assertions on status */
    mset = cudaMemset(data, 0.0, sizeof(double)*size);
    assert(alloc == cudaSuccess and mset == cudaSuccess);

}
void CUDAStorage::_copy(CUDAStorage *b){
    int incx=1, incy=1, status;
    status = cublasDcopy(ctx->handle(), size, b->data, incx, data, incy);
    assert (status == CUBLAS_STATUS_SUCCESS);
}

CUDAStorage::~CUDAStorage(){
    cudaFree(data);
}

CUDAContext* CUDAStorage::ctx = CUDAContext::getInstance();
