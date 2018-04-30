#include "ops.h"
#include "cublas_v2.h"

CUDAContext *ctx = CUDAContext::getInstance();

CUDATensor operator+(CUDATensor A, CUDATensor B){
    CUDATensor C(B.rows, B.cols);
    C = B;

    int n, incx, incy;
    double alpha;

    alpha = 1.0;
    incx = 1, incy = 1;
    n = A._size();


    cublasDaxpy(ctx->handle(), n, &alpha, 
            A.storage->data, incx,
            C.storage->data, incy);
    return C;
}

CUDATensor operator-(CUDATensor A, CUDATensor B){
    CUDATensor C(B.rows, B.cols);
    C = A;

    int n, incx, incy;
    double alpha;

    alpha = -1.0;
    incx = 1, incy = 1;
    n = A._size();


    cublasDaxpy(ctx->handle(), n, &alpha, 
            B.storage->data, incx,
            C.storage->data, incy);
    return C;
}

