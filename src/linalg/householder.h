#include "linalg.h"

CPUTensor reflector(CPUTensor v) {
    /* check if v is 1-dim */
    assert(v.cols == 1);

    /* find norm of the vector */
    double norm_x = norm(v);

    /* subtract it from first element */
    v(0, 0) -= norm_x;

    /* normalize the vector */
    double norm_u = norm(v);

    for(int i = 0; i < v.rows; i++) {
        v(i, 0) /= norm_u;
    }

    return v;
}

CUDATensor reflector(CUDATensor w) {
    CPUTensor v = w.cpu();
    /* check if v is 1-dim */
    assert(v.cols == 1);

    std::cout << v << "\n";

    /* find norm of the vector */

    double norm_x = norm(v.gpu());
    std::cout << "norm: " << norm_x << "\n";

    /* subtract it from first element */
    v(0, 0) -= norm_x;

    /* normalize the vector */
    double norm_u = norm(v);

    for(int i = 0; i < v.rows; i++) {
        v(i, 0) /= norm_u;
    }

    return v.gpu();
}


template <class _Tensor>
_Tensor house(_Tensor v) {
    /* check if v is 1-dim */
    assert(v.cols == 1);

    /* calculate v * v_transpose */
    _Tensor v_transpose = v.transpose();
    /*
    std::cout << v.rows << "x" << v.cols << "\n";
    std::cout << v << "\n";
    std::cout << v_transpose.rows << "x" << v_transpose.cols << "\n";
    std::cout << v_transpose << "\n";
    */

    _Tensor vvT = v*v_transpose;
    
    /* H = I - 2vvT */
    _Tensor I(v.rows, v.rows);
    I = identity(I);
    
    vvT = 2*vvT;
    
    _Tensor H = I - vvT;
    
    return H;
}
