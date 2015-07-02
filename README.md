# Diploma-Thesis ZLAHQRV
ZLAHQRV finds the roots of complex polynomials by calculating eigenvalues of the companion matrix.

## Install Instructions
1. Clone repository
2. Extract testcases.zip in testcases folder
3. Run 'make &lt;target&gt;'  (see Makefile for details)
4. Run './zlahqr' in zlahqrv folder

Further preprocessor options can be added
* -DOPENMP: parallel execution on CPU (OpenMP)
* -DOPENACC: parallel execution on Nvidia GPU (OpenACC)
* -DCUDA: parallel execution on Nvidia GPU (CUDA)
* -DMIC: target for Intel Xeon Phi
* -MKL: use MKL's (Math Kernel Library) version of ZLAHQR
* -DLAPACK: use LAPACK's version of ZLAHQR
* -DLAPACK and -DZGEEV: use LAPACK's version of ZLAHQR, called by LAPACK's ZGEEV

## Licence

The MIT License (MIT)

Copyright (c) 2015 University of Graz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
