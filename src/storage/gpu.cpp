#include "storage.h"

CUDAStorage::CUDAStorage(int size): Storage(size){
    int status;
    status = cudaMalloc((void**)&data, sizeof(double)*size);
    /* TODO assertions on status */
    assert(status == cudaSuccess);

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
