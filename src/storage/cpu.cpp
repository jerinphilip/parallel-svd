#include "storage.h"

CPUStorage::CPUStorage(int size): Storage(size){
    data = (double*)(std::malloc)(sizeof(double)*size);
    memset(data, 0, size);
}

void CPUStorage::_copy(CPUStorage *b){
    assert (size == b->size);
    memcpy(data, b->data, sizeof(double)*size);
}

void CPUStorage::_dcopy(const double *d, int _size){
    assert (size == _size);
    memcpy(data, d, sizeof(double)*size);
}

CPUStorage::~CPUStorage(){
    std::free(data);
    data = NULL;
}
