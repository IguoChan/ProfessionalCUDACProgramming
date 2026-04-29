CU_APPS=checkDimension defineGridBlock sumArraysOnGPU-small-case sumArraysOnGPU-timer \
		checkThreadIndex sumMatrixOnGPU-2D-grid-2D-block checkDeviceInfor \
		sumMatrixOnGPU-2D-grid-1D-block sumMatrixOnGPU-1D-grid-1D-block
C_APPS=sumArraysOnHost

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_89 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}