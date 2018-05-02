#include "linalg.h"

template <class _Tensor>
std::tuple <_Tensor, _Tensor, _Tensor> svd(_Tensor A) {
    bool transpose = false;
    if ( A.rows < A.cols){
        A = A.transpose();
        transpose = true;
    }

    auto bidiag_products = bidiagonalize(A);
    _Tensor Q_t = std::get<0>(bidiag_products);
    _Tensor B = std::get<1>(bidiag_products);
    _Tensor P = std::get<2>(bidiag_products);
    
    _assert(Q_t*A*P, B);
    
    auto diag_products = diagonalize(B);
    _Tensor X_t = std::get<0>(diag_products);
    _Tensor sigma = std::get<1>(diag_products);
    _Tensor Y = std::get<2>(diag_products);
    
    _assert(X_t*B*Y, sigma);
    
    _Tensor Q = Q_t.transpose();
    _Tensor X = X_t.transpose();
    _Tensor U = Q*X;
    
    _Tensor V = P*Y;
    _Tensor V_t = V.transpose();
    
    _assert(U*sigma*V_t, A);

    if (transpose){
        _Tensor tmp;
        tmp = U;
        U = V_t.transpose();
        sigma = sigma.transpose();
        V_t = tmp.transpose();
    }
    
    auto products = std::make_tuple(U, sigma, V_t);
    return products;
}
