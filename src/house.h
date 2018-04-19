#ifndef HOUSE_H
#define HOUSE_H

#include <cassert>

#include "tensor.h"
#include "ops.h"
#include "utils.h"

CPUTensor reflector(CPUTensor v) {
    /* check if v is 1-dim */
    assert(v.cols == 1);

    /* find norm of the vector */
    double norm_x = ops::norm(v);

    /* subtract it from first element */
    v(0, 0) -= norm_x;
    
    /* normalize the vector */
    double norm_u = ops::norm(v);
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
    
    CPUTensor vvT = ops::mul(v, v_transpose);
    
    /* H = I - 2vvT */
    CPUTensor I(v.rows, v.rows);
    I = identity(I);
    
    vvT = ops::smul(vvT, 2);
    
    CPUTensor H = I-vvT;
    
    return H;
}

#endif
