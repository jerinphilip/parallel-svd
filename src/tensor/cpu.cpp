#include "tensor.h"
#define EPS 1e-9

CPUTensor::CPUTensor(int _rows, int _cols):
    Tensor(_rows, _cols) {
        storage = new CPUStorage(_size());
    }

CPUTensor::CPUTensor(): Tensor(0, 0){
    storage = NULL;
}

CPUTensor::CPUTensor(const CPUTensor &B): Tensor(B.rows, B.cols){
    storage = new CPUStorage(_size());
    storage->_copy(B.storage);
}

bool CPUTensor::is_zero(){
    for(int i=0; i < rows; i++){
        for(int j=0; j <cols; j++){
            if (abs((*this)(i, j)) > EPS)
                return false;
        }
    }
    return true;
}

bool CPUTensor::is_diagonal() {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(i != j) {
                if(abs((*this)(i, j)) >  EPS) {
                    return false;
                }
            }
        }
    }
    return true;
}

void CPUTensor::operator=(const CPUTensor A){
    /* 
     * TODO
     * Possible optimization, if dimensions equal, 
     * remove realloc overhead. 
     * */
    if(storage)
        delete storage;
    rows = A.rows;
    cols = A.cols;
    storage = new CPUStorage(_size());
    storage->_copy(A.storage);
}

CPUTensor CPUTensor::flatten(){
    return reshape(rows*cols, 1);
}

CPUTensor CPUTensor::reshape(int new_rows, int new_cols){
    assert( (rows * cols) == (new_rows * new_cols));

    /* Iterate through standard way. cols, then rows; */
    CPUTensor R(new_rows, new_cols);
    R.storage->_copy(storage);
    //        memcpy(R.storage, storage, sizeof(double)*_size());
    return R;
}

/* transposes in place */
void CPUTensor::_transpose() {
    /* create new CPUStorage transposed */
    CPUStorage *transposed;
    transposed = new CPUStorage(_size());

    /* populate transposed */
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            transposed->data[index(i, j, rows)] = storage->data[index(i, j, rows)];
        }
    }

    /* replace storage with transposed */
    delete storage;
    storage = new CPUStorage(_size());
    storage->_copy(transposed);
    delete transposed;

    /* swap rows and cols for tensor */
    int tmp;
    tmp = rows;
    rows = cols;
    cols = tmp;
}

/* returns new transposed CPUTensor */
CPUTensor CPUTensor::transpose() {
    /* create new CPUStorage transposed */
    CPUStorage *transposed;
    transposed = new CPUStorage(_size());

    /* populate transposed */
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            transposed->data[index(j, i, cols)] = storage->data[index(i, j, rows)];
        }
    }

    /* create new CPUTensor, set its storage to transposed */
    CPUTensor T(cols, rows);
    T.storage->_copy(transposed);

    delete transposed;
    return T;
}

double& CPUTensor::operator()(int i, int j){
    return storage->data[index(i, j, rows)];
}

double CPUTensor::operator()(int i, int j) const {
    return storage->data[index(i, j, rows)];
}

CPUTensor::~CPUTensor(){
    delete storage;
}


CPUTensor CPUTensor::from_array(const double *A, int rows, int cols){
    CPUTensor C(rows, cols);
    C.storage->_dcopy(A, rows*cols);
    return C;
}

std::ostream& operator <<(std::ostream &out, const CPUTensor B){
    for(int i=0; i < B.rows; i++){
        for(int j=0; j < B.cols; j++){
            out << B(i, j) << " ";
        }
        out << "\n";
    }
    return out;
}

CUDATensor CPUTensor::gpu() const{
    CUDATensor C(rows, cols);
    int status;

    status = cublasSetMatrix(rows, cols, sizeof(double),
                storage->data, rows, C.storage->data, C.rows);

    assert (status == cudaSuccess);
    return C;
}

