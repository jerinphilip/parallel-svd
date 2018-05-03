#include <cassert>

#include "storage/storage.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "tensor/indexing.h"
#include "utils/utils.h"
#include "linalg/linalg.h"

int main(int argc, char *argv[]){
    int m, n;
    
    /* m >= n */
    m = 5, n = 2;
    assert(m >= n);

    /* Initialization of a CPU Tensor. */
    CPUTensor A(m, n), B(m, n);
    CUDATensor gA(m, n), gB(m, n);


    /* Utility to get random matrices. "utils.h" */
    A = random(m, n);
    B = random(m, n);
    
    gA = A;
    gB = B;

    /* 
     * Block description syntax. 
     * With slice, this enables slicing of the array.
     * block -> "indexing.h"
     * slice -> "ops.h"
     */
    block s = block(1, 3)(0, -1);

    /*
     * //print_m is a macro utility defined in "utils.h"
     * All ops are defined in "ops.h"
     */


    //print_m(A);
    //print_m(slice(A, s));
    //print_m(slice(gA, s));
    _assert(slice(A, s), slice(gA, s).cpu());
    //print_m(B);
    //print_m(A + B);
    //print_m(A - B);
//    //print_m(A * B);
    //print_m(A / B);

    _assert((A + B),  (gA + gB).cpu());
    _assert((A - B),  (gA - gB).cpu());
    //print_m(gA*3.0);
    //print_m(3.0*A);
    _assert((3.0*A), (A*3.0));
    _assert((3.0*A), (gA*3.0).cpu());
//    //print_m(mul(A, B));
//    _assert(A*B, (gA*gB).cpu());
//
//
    
    std::cout << std::endl << "gogol tests" << std::endl;
    _assert(A, A.transpose().transpose());
/*    //print_m(A);
    std::cout << std::endl << "transpose:" << std::endl;
    A._transpose();
    B._transpose();
    //print_m(A);
    //print_m(B);
    //print_m(A+B);
    
    CPUTensor C = A.transpose();
    //print_m(C);
    double test = norm(A);
    std::cout << "norm of A is " << test << std::endl;
    
    CPUTensor house_v = reflector(A.flatten());
    //print_m(house_v);
    
    CPUTensor house_H = house(house_v);
    //print_m(house_H);*/
/*    
    //print_m(A);
    auto bidiag_products = bidiagonalize(A);
    CPUTensor C = std::get<0>(bidiag_products);
    CPUTensor D = std::get<1>(bidiag_products);
    CPUTensor E = std::get<2>(bidiag_products);
    //print_m(C);
    //print_m(D);
    //print_m(E);
    
    //print_m(D);
    //print_m(check_zeros(C*A*E));
    
    _assert(C*A*E, D);
    
    C = C*C.transpose();
    E = E*E.transpose();
    C = check_zeros(C);
    E = check_zeros(E);
    
    //print_m(C);
    //print_m(E);
*//*
    CUDATensor CDA(C); 
    //print_m(CDA);
*//*    
    _assert((A+B)+(A-B), 2*A);
    
    auto diag_products = diagonalize(D);
    CPUTensor P = std::get<0>(diag_products);
    CPUTensor Q = std::get<1>(diag_products);
    CPUTensor R = std::get<2>(diag_products);
    
    _assert(P*D*R, Q);
    //print_m(Q);
    //print_m(check_zeros(P*P.transpose()));
    //print_m(check_zeros(R*R.transpose()));
*/
    auto svd_products = svd(A);
    CPUTensor U = std::get<0>(svd_products);
    CPUTensor sigma = std::get<1>(svd_products);
    CPUTensor V_t = std::get<2>(svd_products);
    
    //print_m(U);
    //print_m(sigma);
    //print_m(V_t.transpose());
    
    //print_m(check_zeros(U*U.transpose()));
    //print_m(check_zeros(V_t*V_t.transpose()));
/*    
    CPUTensor A_t = A.transpose();
    
    //print_m(A);
    //print_m(gA);
    CUDATensor gA_t = transpose(gA);
    CUDATensor gA_t = A_t.gpu();
    A_t = gA_t.cpu();
    A = A.transpose();
    //print_m(A);
    CUDATensor test = transpose(gA_t);
    std::cout << "before printing test\n";
    //print_m(test);
    std::cout << "after printing test\n";
    
    _assert(transpose(gA_t).cpu(), gA.cpu());
*/    
    CPUTensor x(m, 1), y(m, 1);
    CUDATensor X(m, 1), Y(m, 1);
    x = random(m, 1);
    y = random(m, 1);
    X = x;
    Y = y;
    
    double r;
    r = dot(X, Y);
/*    
    //print_m(x);
    //print_m(y);
    //print_m(X);
    //print_m(Y);
    //print_m(r);
*/
    _assert(dot(x, y), dot(X, Y));
    
/*    //print_m(x)
    //print_m(X);
    //print_m(norm(x));
    //print_m(norm(X));*/
    _assert(norm(x), norm(X));
    
    A = random(6, 6);
    gA = A.gpu();
    /*
    A = identity(A);
    gA = identity(gA);
    */
    block b69 = block(1, A.rows)(1, A.cols);
    B = slice(A, b69);
    B = identity(B);
    set_slice(A, b69, B);

    gB = slice(gA, b69);
    gB = identity(gB);
    set_slice(gA, b69, gB);

    _assert(A, gA.cpu());
    print_m(A);
    print_m(gA);
    
    std::cout << "end\n";
    return 0;
}
