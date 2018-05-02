#include "utils.h"
#include "cublas_v2.h"

CUDAContext *ctx2 = CUDAContext::getInstance();

static const char* _UtilsError(cublasStatus_t error)
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

CPUTensor random(int rows, int cols){
    CPUTensor C(rows, cols);
    for(int i=0; i < rows; i++){
        for(int j=0; j < cols; j++){
            C(i, j) = (double)(rand()%10);
        }
    }
    return C;
}

CPUTensor identity(CPUTensor I) {
    /* check if I is square */
    assert(I.rows == I.cols);
    
    /* make it an identity matrix */
    for(int i=0; i < I.rows; i++) {
        for(int j=0; j < I.cols; j++) {
            if(i == j) {
                I(i, j) = 1;
            } else {
                I(i, j) = 0;
            }
        }
    }
    return I;
}

CUDATensor identity(CUDATensor I) {
    assert(I.rows == I.cols);
    CPUTensor i(I.rows, I.cols);
    i = identity(i);
    return i.gpu();
}

/* puts A in the bottom left corner of I matrix */
CPUTensor id_pad(CPUTensor A, int m) {
    /* check if dims of A < m */
    assert(A.rows <= m && A.cols <= m);
    
    /* create identity matrix */
    CPUTensor I(m, m);
    I = identity(I);
    
    /* overwrite with A from bottom-right */
    for(int i = m-1, j = A.rows-1; i >= 0, j >= 0; i--, j--) {
        for(int k = m-1, l = A.cols-1; k >= 0, l >= 0; k--, l--) {
            I(i, k) = A(j, l);
        }
    }
    
    return I;
}

/* puts A's top left corner at the (pos, pos) position */
CPUTensor id_pad_at(int pos, CPUTensor A, int m) {
    /* check if dims of A < m adjusted at pos */
    assert(A.rows <= m-pos && A.cols <= m-pos);
    
    /* create identity matrix */
    CPUTensor I(m, m);
    I = identity(I);
    
    /* overwrite A starting from (pos, pos) */
    for(int i = pos, p = 0; p < A.rows; i++, p++) {
        for(int j = pos, q = 0; q < A.cols; j++, q++) {
            I(i, j) = A(p, q);
        }
    }
    
    return I;
}

CPUTensor check_zeros(CPUTensor A) {
    for(int i = 0; i < A.rows; i++) {
        for(int j = 0; j < A.cols; j++) {
            if(A(i, j) < 0.000000001 && A(i, j) > -0.000000001) {
                A(i, j) = 0;
            }
        }
    }
    return A;
}
