#ifndef OPS_H
#define OPS_H
#include "tensor.h"


/*
Tensor operator+(const Tensor A, const Tensor B){}

Tensor operator-(const Tensor A, const Tensor B){}

Tensor operator*(const Tensor A, const Tensor B){}

Tensor operator*(const double k, const Tensor A){}

Tensor operator*(const Tensor A, const double k){ return k*A; }

tuple<Tensor, Tensor, Tensor> bidiagonalize(const Tensor A){

}

tuple<Tensor, Tensor, Tensor> diagonalize(const Tensor A){

}

tuple<Tensor, Tensor, Tensor> SVD(const Tensor A){

}
*/

#define CPU_ELEMENT_WISE(op) \
    CPUTensor operator op(const CPUTensor A, const CPUTensor B){ \
        /* Assertions */                                         \
        assert (A.rows == B.rows and A.cols == B.cols);          \
        CPUTensor C(A.rows, A.cols);                             \
                                                                 \
        /* Settling for two dimensions now */                    \
        for(int i = 0; i < A.rows; i++){                         \
            for(int j=0; j < A.cols; j++){                       \
                C(i, j) = A(i, j) op B(i, j);                   \
            }                                                    \
        }                                                        \
        return C;                                                \
    }

CPU_ELEMENT_WISE(+)
CPU_ELEMENT_WISE(-)

#endif
