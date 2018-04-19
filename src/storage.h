#ifndef STORAGE_H
#define STORAGE_H

#include <iostream>
#include <vector>
#include <cstring>



struct Storage {
    double *data;
    int size;
    Storage(int size): size(size){}
};

struct CPUStorage : public Storage {
    CPUStorage(int size): Storage(size){
        data = (double*)(malloc)(sizeof(double)*size);
        memset(data, 0, size);
    }

    void _copy(CPUStorage *b){
        memcpy(data, b->data, sizeof(double)*size);
    }

    ~CPUStorage(){
        free(data);
    }

};

struct CUDAStorage: public Storage {

};

#endif
