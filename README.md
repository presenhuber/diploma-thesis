# Diploma-Thesis ZLAHQRV
ZLAHQRV finds the roots of complex polynomials by calculating eigenvalues of the companion matrix.

## Install Instructions
1. Clone repository
2. Extract testcases.zip in testcases folder
3. Run 'make &lt;target&gt;'  (see Makefile for details)
4. Run './zlahqr' in zlahqrv folder

Further Compiler flags can be added
* -DOPENMP: parallel execution on CPU (OpenMP)
* -DOPENACC: parallel execution on Nvidia GPU (OpenACC)
* -DCUDA: parallel execution on Nvidia GPU (CUDA)
* -DMIC: target for Intel Xeon Phi
* -MKL: use MKL's (Math Kernel Library) version of ZLAHQR
* -DLAPACK: use LAPACK's version of ZLAHQR
* -DLAPACK and -DZGEEV: use LAPACK's version of ZLAHQR, called by LAPACK's ZGEEV
