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

    /* find norm of the vector */
    //print_m(v.transpose());

    double norm_x = norm(v.gpu());
    // print_m(norm_x);

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

    
    _Tensor I(v.rows, v.rows);
    I = identity(I);
    
    _Tensor H = I - 2*v*v.transpose();
    
    return H;
}
