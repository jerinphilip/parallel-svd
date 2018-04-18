#ifndef OPS_H
#define OPS_H
#include "tensor.h"


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



#endif
