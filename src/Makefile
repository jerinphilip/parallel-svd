_NVCC='nvcc'
NVFLAGS='-lcublas'
CXXFLAGS='-std=c++11'
DEBUGFLAGS='-g'

NVCC=$(_NVCC) $(NVFLAGS) $(CXXFLAGS) $(DEBUGFLAGS)


test: libtensor.so
	$(NVCC) -Xcompiler -L. -ltensor -lstorage test.cpp

test_cuda:
	$(NVCC) tensor/cpu.cpp tensor/gpu.cpp storage/cpu.cpp storage/gpu.cpp test_cuda.cpp

libstorage.so:
	$(NVCC) -Xcompiler -fPIC  storage/*.cpp -shared -o libstorage.so

libtensor.so:  libstorage.so
	$(NVCC) -Xcompiler -fPIC -L.  -shared -o libtensor.so tensor/*.cpp
