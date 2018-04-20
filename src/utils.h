#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

#include "tensor/tensor.h"
#include "tensor/indexing.h"

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

CPUTensor id_pad(CPUTensor A, int m) {
    /* check if dims of A < m */
    assert(A.rows < m && A.cols < m);
    
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

#define print_m(m) \
    std::cout << #m << ":\n";\
    std::cout << m << "\n" ;

#define _assert(x, y) \
    std::cout << (#x) << " = " << (#y) << "?\n"; \
    assert((x) == (y));


#endif
