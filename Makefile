# //============================================================================
# // Name        : Makefile
# // Author      : M. Presenhuber, M. Liebmann
# // Version     : 1.0
# // Copyright   : University of Graz
# // Description : ZLAQHRV-Algorithm
# //============================================================================

makefile:
all:
gpp:
#	g++ -O3 -march=native -o zlahqr zlahqr.cpp -Wall -lm
	g++ -O3 -march=native -fopenmp -o zlahqr zlahqr.cpp -Wall -lm -DOPENMP
#	g++ -O3 -march=native -fopenmp -o zlahqr zlahqr.cpp -Wall -lm -llapack -DOPENMP -DLAPACK
#	g++ -O3 -march=native -fopenmp -o zlahqr zlahqr.cpp -Wall -lm -llapack -DOPENMP -DLAPACK -DZGEEV

cuda:
	nvcc -O3 -x cu -arch=sm_20 -fmad=false -o zlahqr zlahqr.cpp -lm -DCUDA

kepler:
	nvcc -O3 -x cu -arch=sm_35 -o zlahqr zlahqr.cpp -lm -DCUDA
	
icpc:
	icpc -O3 -mavx -openmp -o zlahqr zlahqr.cpp -Wall -fp-model precise -lm -DOPENMP
	
pgcpp:
	pgc++ -O4 -acc -Minfo=accel -ta=tesla:cc20,cc35 -o zlahqr zlahqr.cpp -lm -DOPENACC

mkl:
	icpc -O3 -march=native -openmp -o zlahqr zlahqr.cpp -Wall -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -DOPENMP -DMKL
	
micoff:
	icpc -O3 -march=native -openmp -o zlahqr zlahqr.cpp -Wall -lm -offload-attribute-target=mic -DMIC -DOPENMP

mic:
	icpc -O3 -mmic -openmp -o zlahqr zlahqr.cpp -Wall -lm -DOPENMP
	
