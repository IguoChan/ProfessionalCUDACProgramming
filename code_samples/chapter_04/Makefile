CU_APPS=globalVariable memTransfer pinMemTransfer readSegment simpleMathAoS simpleMathSoA sumArrayZerocpy writeSegment
READSEGMENT_CACHE_APPS=readSegment_ca readSegment_cg
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_89 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
readSegment_ca: readSegment.cu
	nvcc -O2 -arch=sm_89 -Xptxas -dlcm=ca -o $@ $<
readSegment_cg: readSegment.cu
	nvcc -O2 -arch=sm_89 -Xptxas -dlcm=cg -o $@ $<
readSegment-cache: ${READSEGMENT_CACHE_APPS}
clean:
	rm -f ${CU_APPS} ${C_APPS} ${READSEGMENT_CACHE_APPS} readSegment_default
