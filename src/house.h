#ifndef HOUSE_H
#define HOUSE_H

#include <cassert>

#include "tensor.h"
#include "ops.h"

CPUTensor reflector(CPUTensor v) {
    /* check if v is 1-dim */
    assert(v.cols == 1);

    /* find norm of the vector */
    double norm_x = ops::norm(v);
    print_m(norm_x);
    print_m(v);
    
    /* subtract it from first element */
    v(0, 0) -= norm_x;
    print_m(v);
    
    /* normalize the vector */
    double norm_u = ops::norm(v);
    for(int i = 0; i < v.rows; i++) {
        v(i, 0) /= norm_u;
    }
    print_m(norm_u);
    print_m(v);
    
    return v;
}

#endif
