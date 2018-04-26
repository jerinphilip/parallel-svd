#include "linalg.h"

std::tuple <CPUTensor, CPUTensor, CPUTensor> svd(CPUTensor A) {
    auto bidiag_products = bidiagonalize(A);
    CPUTensor Q_t = std::get<0>(bidiag_products);
    CPUTensor B = std::get<1>(bidiag_products);
    CPUTensor P = std::get<2>(bidiag_products);
    
    _assert(Q_t*A*P, B);
    
    auto diag_products = diagonalize(B);
    CPUTensor X_t = std::get<0>(diag_products);
    CPUTensor sigma = std::get<1>(diag_products);
    CPUTensor Y = std::get<2>(diag_products);
    
    _assert(X_t*B*Y, sigma);
    
    CPUTensor Q = Q_t.transpose();
    CPUTensor X = X_t.transpose();
    CPUTensor U = Q*X;
    
    CPUTensor V = P*Y;
    CPUTensor V_t = V.transpose();
    
    _assert(U*sigma*V_t, A);
    
    auto products = std::make_tuple(U, sigma, V_t);
    return products;
}
