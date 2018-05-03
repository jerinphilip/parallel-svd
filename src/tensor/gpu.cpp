#include "tensor.h"

CUDATensor::CUDATensor(int _rows, int _cols): Tensor(_rows, _cols){
    storage = new CUDAStorage(_size());
}

CUDATensor::CUDATensor(): Tensor(0, 0) {
    storage = NULL;
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

CUDATensor::CUDATensor(const CUDATensor &B): Tensor(B.rows, B.cols){
    storage = new CUDAStorage(_size());
    storage->_copy(B.storage);
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

CUDATensor CUDATensor::reshape(int new_rows, int new_cols){
    assert( (rows * cols) == (new_rows * new_cols));
    CUDATensor R(new_rows, new_cols);
    R.storage->_copy(storage);
    //        memcpy(R.storage, storage, sizeof(double)*_size());
    return R;

}

CUDATensor CUDATensor::flatten(){
    return reshape(rows*cols, 1);
}

std::ostream& operator <<(std::ostream &out, const CUDATensor B){
    out << B.cpu();
    return out;
}

CUDATensor::~CUDATensor(){
    delete storage;
}

bool CUDATensor::is_diagonal(){
    CPUTensor C = cpu();
    return C.is_diagonal();
}

CUDATensor CUDATensor::from_array(double *A, int rows, int cols){
    CUDATensor C(rows, cols);
    int status;

    status = cublasSetMatrix(rows, cols, sizeof(double), 
            A,
            rows, 
            C.storage->data, rows);
    assert(status == CUBLAS_STATUS_SUCCESS);
    return C;
}
