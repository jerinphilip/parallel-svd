#ifndef STORAGE_H
#define STORAGE_H
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "cuda.h"

struct Storage;
struct CPUStorage;
struct CUDAStorage;

struct Storage {
    double *data;
    int size;
    Storage(int size): size(size){}
};

struct CPUStorage : public Storage {
    CPUStorage(int size);
    void _copy(CPUStorage *b);
    void _dcopy(const double *d, int _size);
    ~CPUStorage();
};

struct CUDAStorage: public Storage {
    static CUDAContext *ctx;
    CUDAStorage(int size);
    void _copy(CUDAStorage *b);
    ~CUDAStorage();
};

#endif
