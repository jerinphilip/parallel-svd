#ifndef LINALG_H
#define LINALG_H

#include <cassert>
#include <tuple>

#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "tensor/indexing.h"
#include "utils/utils.h"

/* Householder and Givens stuff */

template <class _Tensor>
_Tensor reflector(_Tensor);

template <class _Tensor>
_Tensor house(_Tensor);

template <class _Tensor>
_Tensor givens(_Tensor, int, int);

template <class _Tensor>
std::tuple <_Tensor, _Tensor, _Tensor> bidiagonalize(_Tensor);

template <class _Tensor>
std::tuple <_Tensor, _Tensor, _Tensor> diagonalize(_Tensor);

template <class _Tensor>
std::tuple <_Tensor, _Tensor, _Tensor> svd(_Tensor A);

#include "householder.h"
#include "givens.h"
#include "bidiagonalize.h"
#include "diagonalize.h"
#include "svd.h"

#endif
