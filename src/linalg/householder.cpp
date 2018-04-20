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

CPUTensor house(CPUTensor v) {
    /* check if v is 1-dim */
    assert(v.cols == 1);

    /* calculate v * v_transpose */
    CPUTensor v_transpose = v.transpose();
    
    CPUTensor vvT = v*v_transpose;
    
    /* H = I - 2vvT */
    CPUTensor I(v.rows, v.rows);
    I = identity(I);
    
    vvT = 2*vvT;
    
    CPUTensor H = I-vvT;
    
    return H;
}
