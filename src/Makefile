_NVCC='nvcc'
NVFLAGS='-lcublas'
CXXFLAGS='-std=c++11'
DEBUGFLAGS='-g'

NVCC=$(_NVCC) $(NVFLAGS) $(CXXFLAGS) $(DEBUGFLAGS)


test: build/libtensor.so build/libstorage.so build/liblinalg.so build/libutils.so
	$(NVCC) -Xcompiler -L./build/ -ltensor -lstorage -llinalg -lutils test.cpp

build/libstorage.so:
	$(NVCC) -Xcompiler -fPIC  storage/*.cpp -shared -o build/libstorage.so

build/libtensor.so:  build/libstorage.so
	$(NVCC) -Xcompiler -fPIC -L./build/  -shared -o build/libtensor.so tensor/*.cpp

build/liblinalg.so: build/libtensor.so
	$(NVCC) -Xcompiler -fPIC -L./build  -shared -o build/liblinalg.so linalg/*.cpp

build/libutils.so: build/libtensor.so
	$(NVCC) -Xcompiler -fPIC -L./build -shared -o build/libutils.so utils/*.cpp
