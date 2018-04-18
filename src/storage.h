#ifndef STORAGE_H
#define STORAGE_H

#include <iostream>
#include <vector>



struct Storage {
    double *data;
    int size;
    std::vector<int> dims;
    Storage(int size): size(size){}
};

struct CPUStorage : public Storage {
    CPUStorage(int size): Storage(size){
        data = (double*)(malloc)(sizeof(double)*size);
    }

    ~CPUStorage(){
        free(data);
    }

};

struct CUDAStorage: public Storage {

};

#endif
