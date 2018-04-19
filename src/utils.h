#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

#include "tensor.h"
#include "indexing.h"

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

#define print_m(m) \
    std::cout << #m << ":\n";\
    std::cout << m ;

#endif
