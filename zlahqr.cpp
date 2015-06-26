//============================================================================
// Name        : zlahqr.cpp
// Author      : M. Presenhuber, M. Liebmann
// Version     : 1.0
// Copyright   : University of Graz
// Description : ZLAQHRV-Algorithm
//============================================================================

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <sys/time.h>

#ifdef OPENMP
#include "omp.h"
#endif

#ifdef OPENACC
#include "openacc.h"
#endif

#ifdef CUDA
#define cuda __host__ __device__
#define global __global__
#else
#define cuda
#define global
#endif

#ifdef LAPACK
#define ZLAHQR zlahqr__
#else
#define ZLAHQR zlahqr_
#endif

#include "Toolbox.hpp"

using namespace std;

void loadPackages(int packages, int matrices, int n_coeffs, ComplexDouble* C)
{
    char numchar[3] = "";
    string numstring;
    string filePath;

    for (int package = 0; package < packages; package++)
    {
        sprintf(numchar, "%d", package + 1);
        string numstring(numchar);
        filePath = "testcases/f_coeffs_" + numstring + ".csv";
        cout << "loading package: " << filePath << endl;
        ifstream filestream(filePath.c_str());
        string line;

        for (int lineNumber = 0; lineNumber < matrices; lineNumber++)
        {
            getline(filestream, line);
            stringstream lineStream(line);
            string cell;
            double coeffs[n_coeffs * 2];
            for (int coeffNumber = 0; coeffNumber < n_coeffs * 2; coeffNumber++)
            {
                getline(lineStream, cell, ',');
                coeffs[coeffNumber] = atof(cell.c_str());
            }
            for (int coeff_n = 0; coeff_n < n_coeffs; coeff_n++)
            {
                C[coeff_n + lineNumber * n_coeffs + package * matrices * n_coeffs] = ComplexDouble(coeffs[coeff_n * 2], coeffs[coeff_n * 2 + 1]);
            }
        }
    }
}

int main(int argc, char** argv)
{
    int packages = 3;
    int lines = 147456; //147456

#ifdef CUDA
    int block_cnt = 576 * packages;
    int thread_cnt = 256; // CPU:16 GPU:256 ACC:32 MIC:16
#else

#ifdef OPENACC
    int block_cnt = 576 * packages;
    int thread_cnt = 256; // CPU:16 GPU:256 ACC:32 MIC:16
#else
    int block_cnt = 18432 * packages;
    int thread_cnt = 8; // CPU:16 GPU:256 ACC:32 MIC:16
#endif

#endif

    int n_coeffs = 11;
    int n_matrices = packages * lines;
    int matrix_dim = n_coeffs - 1;
    int matrix_dim1 = n_coeffs;
    int matrix_dim2 = matrix_dim * matrix_dim;

    if (block_cnt * thread_cnt != n_matrices)
    {
        cout << "Parameter mismatch ..." << block_cnt * thread_cnt << " : " << n_matrices << endl;
        return 0;
    }

    ComplexDouble* C = new ComplexDouble[n_matrices * matrix_dim1]();
    ComplexDouble* E = new ComplexDouble[n_matrices * matrix_dim]();
    double* rC = new double[block_cnt * thread_cnt * matrix_dim1]();
    double* iC = new double[block_cnt * thread_cnt * matrix_dim1]();
    double* rE = new double[block_cnt * thread_cnt * matrix_dim]();
    double* iE = new double[block_cnt * thread_cnt * matrix_dim]();
    double* realVector = new double[block_cnt * thread_cnt * matrix_dim2]();
    double* imagVector = new double[block_cnt * thread_cnt * matrix_dim2]();

    cout << "Loading Packages ..." << endl;

    loadPackages(packages, lines, n_coeffs, C);

    for (int block = 0, i = 0; block < block_cnt; block++)
    {
        for (int thread = 0; thread < thread_cnt; thread++)
        {
            for (int dim = 0; dim < matrix_dim1; dim++, i++)
            {
                rC[thread + dim * thread_cnt + block * thread_cnt * matrix_dim1] = C[i].r;
                iC[thread + dim * thread_cnt + block * thread_cnt * matrix_dim1] = C[i].i;
            }
        }
    }

    cout << "Loading Packages complete." << endl;

#ifdef CUDA
    cudaFree(0);

    double *rC_ = 0, *iC_ = 0, *rE_ = 0, *iE_ = 0;

    cudaMalloc(&rC_, n_matrices * matrix_dim1 * sizeof(double));
    cudaMalloc(&iC_, n_matrices * matrix_dim1 * sizeof(double));
    cudaMalloc(&rE_, n_matrices * matrix_dim * sizeof(double));
    cudaMalloc(&iE_, n_matrices * matrix_dim * sizeof(double));

    double *realVector_ = 0, *imagVector_ = 0;

    cudaMalloc(&realVector_, block_cnt * thread_cnt * matrix_dim2 * sizeof(double));
    cudaMalloc(&imagVector_, block_cnt * thread_cnt * matrix_dim2 * sizeof(double));

    cout << "CUDA Device Memory: " << 16.0 * (matrix_dim1 + matrix_dim + matrix_dim2) * block_cnt * thread_cnt / 1024 / 1024 / 1024 << " GB" << endl;

    cudaFuncSetCacheConfig(zlahqr, cudaFuncCachePreferL1);
#endif

#ifdef OPENACC
    acc_init(acc_device_nvidia);
#endif

    cout << "Calculating Eigenvalues ..." << endl;

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

#ifdef CUDA
    cudaMemcpy(rC_, rC, n_matrices * matrix_dim1 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(iC_, iC, n_matrices * matrix_dim1 * sizeof(double), cudaMemcpyHostToDevice);
    zlahqr<<<block_cnt, thread_cnt>>>(rC_, iC_, rE_, iE_, realVector_, imagVector_, matrix_dim, matrix_dim1, matrix_dim2, block_cnt, thread_cnt);
    cudaMemcpy(rE, rE_, n_matrices * matrix_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(iE, iE_, n_matrices * matrix_dim * sizeof(double), cudaMemcpyDeviceToHost);
#else
    zlahqr(rC, iC, rE, iE, realVector, imagVector, matrix_dim, matrix_dim1, matrix_dim2, block_cnt, thread_cnt);
#endif

    gettimeofday(&t2, NULL);
    double cgpu = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 0.000001;
    cout << "Calculating complete. " << cgpu << " seconds" << endl;

    for (int block = 0, i = 0; block < block_cnt; block++)
    {
        for (int thread = 0; thread < thread_cnt; thread++)
        {
            for (int dim = 0; dim < matrix_dim; dim++, i++)
            {
                E[i].r = rE[thread + dim * thread_cnt + block * thread_cnt * matrix_dim];
                E[i].i = iE[thread + dim * thread_cnt + block * thread_cnt * matrix_dim];
            }
        }
    }

    delete[] C;
    delete[] E;
    delete[] rC;
    delete[] iC;
    delete[] rE;
    delete[] iE;
    delete[] realVector;
    delete[] imagVector;

    return 0;
}
