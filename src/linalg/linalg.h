#ifndef LINALG_H
#define LINALG_H

#include <cassert>
#include <tuple>

#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "tensor/indexing.h"
#include "utils/utils.h"

/* Householder stuff */

CPUTensor reflector(CPUTensor);
CPUTensor house(CPUTensor);
std::tuple <CPUTensor, CPUTensor, CPUTensor> bidiagonalize(CPUTensor);

#endif
