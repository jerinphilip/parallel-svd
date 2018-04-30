#include "tensor.h"
#include "cublas_v2.h"
//#include "utils/utils.h"

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

CUDAContext *ctx2 = CUDAContext::getInstance();

CUDATensor::CUDATensor(int _rows, int _cols): Tensor(_rows, _cols){
    storage = new CUDAStorage(_size());
}

CUDATensor::CUDATensor(CPUTensor C): Tensor(C.rows, C.cols){
    storage = new CUDAStorage(_size());
    int status;
    status = cublasSetMatrix(rows, cols, sizeof(double), 
            C.storage->data,
            rows, 
            storage->data, rows);
    assert(status == CUBLAS_STATUS_SUCCESS);
}

void CUDATensor::_copy(CUDATensor *B){
    storage->_copy(B->storage);
}

void CUDATensor::operator=(const CUDATensor &B){
    if(storage)
        delete storage;

    rows = B.rows;
    cols = B.cols;
    storage = new CUDAStorage(_size());
    storage->_copy(B.storage);
} 


CPUTensor CUDATensor::cpu() const{
    double *buffer;
    buffer = NULL;
    buffer = (double*)std::malloc(_size()*sizeof(double));
    int status;
    status = cublasGetMatrix(rows, cols, 
            sizeof(double), storage->data, 
            rows, 
            buffer, rows);
    assert(status == CUBLAS_STATUS_SUCCESS);
    CPUTensor C = CPUTensor::from_array(buffer, rows, cols);
    std::free(buffer);
    return C;
}

std::ostream& operator <<(std::ostream &out, const CUDATensor B){
    out << B.cpu();
    return out;
}
