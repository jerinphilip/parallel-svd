_NVCC='nvcc'
NVFLAGS='-lcublas'
CXXFLAGS='-std=c++11'
DEBUGFLAGS='-g'

NVCC=$(_NVCC) $(NVFLAGS) $(CXXFLAGS) $(DEBUGFLAGS)

test:
	$(NVCC) tensor/cpu.cpp tensor/gpu.cpp storage/cpu.cpp storage/gpu.cpp storage/cuda.cpp test.cpp

test_cuda:
	$(NVCC) tensor/cpu.cpp tensor/gpu.cpp storage/cpu.cpp storage/gpu.cpp test_cuda.cpp
