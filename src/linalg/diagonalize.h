#include "linalg.h"

std::tuple<CPUTensor, CPUTensor, CPUTensor> diagonalize(CPUTensor B) {
    CPUTensor X_t(B.rows, B.rows);
    CPUTensor sigma(B.rows, B.cols);
    CPUTensor Y(B.cols, B.cols);
    
    CPUTensor x(0, 0);
    CPUTensor y(0, 0);
    CPUTensor G(0, 0);
    
    X_t = identity(X_t);
    Y = identity(Y);
    
    /* find length of upper diagonal */
    int udiag;
    if (B.rows < B.cols) {
        udiag = B.rows;
    } else {
        udiag = B.cols-1;
    }
    
    while(!B.is_diagonal()) {
        /* iterate for each pair of diagonal and upper diagonal elements to 
           eliminate the upper diagonal element                               */
        for(int i = 0; i < udiag; i++) {
            /* slice the row pair to be rotated out of B */
            block pair1 = block(i, i+1)(i, i+2);
            x = slice(B, pair1);
            
            /* do givens rotation based on x */
            G = givens(x.transpose(), i, B.cols);
            
            B = B*G.transpose();
            Y = Y*G.transpose();

            /* slice the col pair to be rotated out of B */
            block pair2 = block(i, i+2)(i, i+1);
            y = slice(B, pair2);
            
            /* do givens rotation based on y */
            G = givens(y, i, B.rows);
            
            B = G*B;
            X_t = G*X_t;
            
            B = check_zeros(B);            
            if(B.is_diagonal()) {
                break;
            }
        }
    }
    sigma = B;
    
    auto products = std::make_tuple(X_t, sigma, Y);
    
    return products;
}
