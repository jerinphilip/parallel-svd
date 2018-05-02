#include "linalg.h"

/* takes 1x2 column vector x and returns givens matrix to be multiplied from 
   left, takes diagonal location of pair to be rotated as parameter, as well as
   size of givens matrix */
CPUTensor givens(CPUTensor x, int pos, int size) {
    double x1, x2;
    x1 = x(0, 0);
    x2 = x(1, 0);
    
    double xnorm;
    xnorm = norm(x);
    
    CPUTensor G(2, 2);
    G(0, 0) = x1/xnorm;
    G(0, 1) = x2/xnorm;
    G(1, 0) = (-1*x2)/xnorm;
    G(1, 1) = x1/xnorm;
    
    return G;
}   

CUDATensor givens(CUDATensor x, int pos, int size){
    CPUTensor C = x.cpu();
    CPUTensor G = givens(C, pos, size);
    return G.gpu();
}

