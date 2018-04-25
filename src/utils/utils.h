#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

#include "tensor/tensor.h"
#include "tensor/indexing.h"

#define print_m(m) \
    std::cout << #m << ":\n";\
    std::cout << m << "\n" ;

#define _assert(x, y) \
    std::cout << (#x) << " = " << (#y) << "?\n"; \
    assert((x) == (y));

CPUTensor random(int, int);
CPUTensor identity(CPUTensor);
CPUTensor id_pad(CPUTensor,int);
CPUTensor id_pad_at(int, CPUTensor, int);
CPUTensor check_zeros(CPUTensor);
#endif
