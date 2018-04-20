#include "cuda.h"
#include "storage.h"

int main(){
    CUDAStorage *x = new CUDAStorage(20);
    delete x;
    return 0;
}
