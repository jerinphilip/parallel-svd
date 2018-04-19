#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <ctime>
#include <cstdlib>
#include "tensor.h"

CPUTensor random(int rows, int cols){
    CPUTensor C(rows, cols);
    for(int i=0; i < rows; i++){
        for(int j=0; j < cols; j++){
            C(i, j) = (double)rand();
        }
    }
    return C;
}

#define print_m(m) \
    std::cout << #m << ":\n";\
    std::cout << m ;

#endif
