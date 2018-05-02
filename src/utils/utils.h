#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

#include "tensor/tensor.h"
#include "tensor/indexing.h"
#include <chrono>

#define print_m(m) \
    std::cout << #m << ":\n";\
std::cout << m << "\n" ;

#define _assert(x, y) \
    std::cout << (#x) << " = " << (#y) << "? "; \
    assert((x) == (y)); \
    std::cout << "\033[42;30m OK \033[0m\n"; \

CPUTensor random(int, int);
CPUTensor identity(CPUTensor);
CUDATensor identity(CUDATensor);
CPUTensor id_pad(CPUTensor,int);
CPUTensor id_pad_at(int, CPUTensor, int);
CPUTensor check_zeros(CPUTensor);


template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
        static typename TimeT::rep execution(F&& func, Args&&... args)
        {
            auto start = std::chrono::steady_clock::now();
            std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
            auto duration = std::chrono::duration_cast< TimeT> 
                (std::chrono::steady_clock::now() - start);
            return duration.count();
        }
};


#endif
