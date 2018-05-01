#include "ops.h"
#include "cublas_v2.h"

CUDAContext *ctx = CUDAContext::getInstance();

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

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

CUDATensor operator*(double alpha, CUDATensor A){
    CUDATensor C(A.rows, A.cols);
    int n, incx, incy;
    incx = 1, incy = 1;
    n = A._size();
    cublasDaxpy(ctx->handle(), n, &alpha, 
            A.storage->data, incx,
            C.storage->data, incy);
    return C;
}

CUDATensor operator*(CUDATensor A, double alpha){
    return alpha*A;
}

CUDATensor operator*(CUDATensor A, CUDATensor B){
    assert (A.cols == B.rows);
    CUDATensor C(A.rows, B.cols);

    double alpha, beta;
    beta = 0.0;
    alpha = 1.0;

    cublasDgemm(ctx->handle(), 
            CUBLAS_OP_N, CUBLAS_OP_N,
            A.rows, B.cols, A.rows,
            &alpha,
            A.storage->data, A.rows,
            B.storage->data, B.rows,
            &beta,
            C.storage->data, C.rows);

    return C;
}

CUDATensor transpose(CUDATensor A) {
    CUDATensor T(A.cols, A.rows);
    
    double alpha, beta;
    alpha = 1.0;
    
    cublasStatus_t status;
    status = cublasDgeam(ctx->handle(),
                CUBLAS_OP_T, CUBLAS_OP_N, 
                T.rows, T.cols,
                &alpha,
                A.storage->data, A.rows, 
                &beta,
                A.storage->data, A.rows,
                T.storage->data, T.rows);
//    std::cout << _cudaGetErrorEnum(status) << std::endl;
    assert(status == CUBLAS_STATUS_SUCCESS);  
    return T;
}

double norm2(CUDATensor A) {
    assert(A.cols == 1);
 
    double result;
    int incx;
    incx = 1;
    cublasStatus_t status;   
    status = cublasDnrm2(ctx->handle(), A.rows, A.storage->data, incx, &result);
    assert(status == CUBLAS_STATUS_SUCCESS);
    return result;
}

double norm(CUDATensor A) {
    CUDATensor flattened = A.flatten();
    double n2 = norm2(flattened);
    double n = sqrt(n2);
    return n;
}

double dot(CUDATensor A, CUDATensor B) {
    assert(A.rows == B.rows);
    assert(A.cols == 1 && B.cols == 1);
    
    double result;
    int incx, incy;
    incx = incy = 1;
    
    cublasStatus_t status;
    status = cublasDdot(ctx->handle(), A.rows,
                        A.storage->data, incx,
                        B.storage->data, incy,
                        &result);
    
    return result;                        
}

CUDATensor slice(const CUDATensor A, block b){
    if ( b.row.end == -1) b.row.end = A.rows;
    if ( b.col.end == -1) b.col.end = A.cols;

    if ( b.row.start == -1) b.row.start = 0;
    if ( b.col.start == -1) b.col.start = 0;

    CUDATensor C(b.row.size(), b.col.size());

    /* Column major storage */
    int i, k;
    int incx = 1, incy = 1;
    int status;

    for(int j=b.col.start; j<b.col.end; j++){
        i = index(b.row.start, j, A.rows);
        k = index(0, j-b.col.start, b.row.size());
        // std::cout << "\ni: " << i << " k: " << k << "\n";
        status = cublasDcopy(ctx->handle(), b.row.size(),
                &A.storage->data[i], incx,
                &C.storage->data[k], incy);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    return C;

}
