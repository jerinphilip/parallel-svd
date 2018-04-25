#ifndef LINALG_H
#define LINALG_H

#include <cassert>
#include <tuple>

#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "tensor/indexing.h"
#include "utils/utils.h"

/* Householder and Givens stuff */

CPUTensor reflector(CPUTensor);
CPUTensor house(CPUTensor);
CPUTensor givens(CPUTensor, int, int);
std::tuple <CPUTensor, CPUTensor, CPUTensor> bidiagonalize(CPUTensor);
std::tuple <CPUTensor, CPUTensor, CPUTensor> diagonalize(CPUTensor);

#endif
