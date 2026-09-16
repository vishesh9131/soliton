# One shared library: D core + CUDA kernels. `make` then `pytest tests`.
CUDA_HOME ?= $(CONDA_PREFIX)
NCCL_HOME ?= $(CONDA_PREFIX)
CUDA_ARCH ?= sm_86
CUDA_INC := $(CUDA_HOME)/targets/x86_64-linux/include
CUDA_LIB := $(CUDA_HOME)/targets/x86_64-linux/lib
D_SRC := $(wildcard core/soliton/*.d)
OUT := python/soliton/libsoliton.so

$(OUT): $(D_SRC) build/kernels.o
	ldc2 -O3 -release -mcpu=native -shared -relocation-model=pic -Icore -of=$@.tmp $(D_SRC) build/kernels.o \
		-L-L$(CUDA_LIB) -L-L$(NCCL_HOME)/lib -L-lcudart -L-lcublas -L-lnccl -L-lnvrtc -L-lcuda -L-lstdc++ \
		-L-rpath=$(CUDA_LIB) -L-rpath=$(NCCL_HOME)/lib
	mv $@.tmp $@  # atomic swap: running processes keep the old library

build/kernels.o: core/cuda/kernels.cu
	@mkdir -p build
	nvcc -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fPIC -I$(CUDA_INC) -I$(NCCL_HOME)/include -c $< -o $@

clean:
	rm -rf build $(OUT)

.PHONY: clean
