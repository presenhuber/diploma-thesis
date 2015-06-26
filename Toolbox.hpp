//============================================================================
// Name        : Toolbox.hpp
// Author      : M. Presenhuber, M. Liebmann
// Version     : 1.0
// Copyright   : University of Graz
// Description : ZLAQHRV-Algorithm
//============================================================================

#ifndef TOOLBOX_HPP_
#define TOOLBOX_HPP_

#include "ComplexDouble.hpp"

cuda void scale_rc(double *rV, double *iV, double zr, double zi, int I, int I1, int I2, int dim, int n)
{
    int off = 1 + dim, j;
    j = (I + (I + 1) * dim - off) * n;
    for (int i = 0; i < I2 - I; ++i)
    {
        cmul(rV[j], iV[j], zr, zi);
        j += dim * n;
    }
    j = (I1 + I * dim - off) * n;
    for (int i = 0; i < I - I1; ++i)
    {
        cmulc(rV[j], iV[j], zr, zi);
        j += n;
    }
}

cuda void single_shift_qr(double *rV, double *iV, int I, int L, int M, int I1, int I2, double v0r, double v0i, double v1r, double v1i, int dim, int n_matrices)
{
    int off = n_matrices + dim * n_matrices;
    double t1r, t1i, t2;
    double h1r, h1i, h2r, h2i, zr, zi, d;
    int i1, i2, i3;

    for (int K = M; K <= I - 1; ++K)
    {
        if (K > M)
        {
            i1 = (K + (K - 1) * dim) * n_matrices - off;
            i2 = i1 + n_matrices;
            v0r = rV[i1];
            v0i = iV[i1];
            v1r = rV[i2];
            v1i = iV[i2];
        }

        if (v1r == 0.0 && v1i == 0.0 && v0i == 0.0)
        {
            t1r = 0.0;
            t1i = 0.0;
        }
        else
        {
            double norm, beta, rsafmn, safmin;
            safmin = DBL_MIN / (DBL_EPSILON * 0.5);

#ifdef MKL
            norm = dabs3(v0r, v0i, dabs2(v1r, v1i));
#else
            norm = sqrt(v0r * v0r + v0i * v0i + v1r * v1r + v1i * v1i);
#endif

            int KNT = 0;
            if (norm < safmin)
            {
                rsafmn = 1.0 / safmin;
                do
                {
                    KNT = KNT + 1;
                    cmul(v1r, v1i, rsafmn);
                    cmul(v0r, v0i, rsafmn);
                    norm *= rsafmn;
                }
                while (norm < safmin);

#ifdef MKL
                norm = dabs3(v0r, v0i, dabs2(v1r, v1i));
#else
                norm = sqrt(v0r * v0r + v0i * v0i + v1r * v1r + v1i * v1i);
#endif

            }
            beta = copysign(norm, v0r);
            zr = v0r + beta;
            zi = v0i;
            t1r = zr;
            t1i = zi;

#ifdef MKL
            cdivex(1.0, 0.0, t1r, t1i, zr, zi);
#else
            cinv(zr, zi);
#endif

            cmul(v1r, v1i, zr, zi);
            cdiv(t1r, t1i, beta);

            for (int i = 0; i < KNT; ++i)
                beta *= safmin;
            v0r = -beta;
            v0i = 0.0;
        }

        if (K > M)
        {
            i1 = (K + (K - 1) * dim) * n_matrices - off;
            i2 = i1 + n_matrices;
            rV[i1] = v0r;
            iV[i1] = v0i;
            rV[i2] = 0.0;
            iV[i2] = 0.0;
        }

        t2 = t1r * v1r - t1i * v1i;

        i1 = (K + K * dim) * n_matrices - off;
        i2 = i1 + n_matrices;
        for (int j = K; j <= I2; ++j)
        {
            h1r = rV[i1];
            h1i = iV[i1];
            h2r = rV[i2];
            h2i = iV[i2];
            zr = h1r;
            zi = h1i;
            cmulc(zr, zi, t1r, t1i);
            cadd(zr, zi, h2r * t2, h2i * t2);
            csub(h1r, h1i, zr, zi);
            cmul(zr, zi, v1r, v1i);
            csub(h2r, h2i, zr, zi);
            rV[i1] = h1r;
            iV[i1] = h1i;
            rV[i2] = h2r;
            iV[i2] = h2i;
            i1 += dim * n_matrices;
            i2 += dim * n_matrices;
        }

        i1 = (I1 + K * dim) * n_matrices - off;
        i2 = i1 + dim * n_matrices;
        i3 = K + 2 < I ? K + 2 : I;
        for (int j = I1; j <= i3; ++j)
        {
            h1r = rV[i1];
            h1i = iV[i1];
            h2r = rV[i2];
            h2i = iV[i2];
            zr = h1r;
            zi = h1i;
            cmul(zr, zi, t1r, t1i);
            cadd(zr, zi, h2r * t2, h2i * t2);
            csub(h1r, h1i, zr, zi);
            cmulc(zr, zi, v1r, v1i);
            csub(h2r, h2i, zr, zi);
            rV[i1] = h1r;
            iV[i1] = h1i;
            rV[i2] = h2r;
            iV[i2] = h2i;
            i1 += n_matrices;
            i2 += n_matrices;
        }
    }

    i1 = (I + (I - 1) * dim) * n_matrices - off;
    zi = iV[i1];
    if (zi != 0.0)
    {
        zr = rV[i1];
        d = cabs(zr, zi);
        cdiv(zr, zi, d);
        zi = -zi;
        rV[i1] = d;
        iV[i1] = 0.0;
        scale_rc(rV, iV, zr, zi, I, I1, I2, dim, n_matrices);
    }

}

#ifdef LAPACK

extern "C" void zgeev_( char* jobvl, char* jobvr, int* n, double* a,
        int* lda, double* w, double* vl, int* ldvl, double* vr, int* ldvr,
        double* work, int* lwork, double* rwork, int* info );

extern "C" void zlahqr_(int* WANTT, int* WANTZ, int* N, int* ILO, int* IHI, double* H, int* LDH, double* W, int* ILOZ, int* IHIZ, double* Z, int* LDZ, int* INFO);

void zlahqr__(double* rC, double* iC, double* rE, double* iE, double* rV, double* iV, int dim, int block_cnt, int thread_cnt)
{
    int n_matrices = thread_cnt;
    double *H = new double[2 * dim * dim];
    double *W = new double[2 * dim];
    double *Z = 0;

    for (int i = 0; i < dim * dim - dim; i++)
    {
        H[2*i+0] = 0.0;
        H[2*i+1] = 0.0;
    }

    double cr = rC[dim * n_matrices];
    double ci = iC[dim * n_matrices];

    int j = (dim * dim - dim);
    for (int i = 0; i < dim; i++)
    {
        double a = rC[i * n_matrices];
        double b = iC[i * n_matrices];
        cdiv(a, b, cr, ci);
        H[2*(j + i)+0] = -a;
        H[2*(j + i)+1] = -b;
    }

    j = 1;
    for (int i = 0; i < dim - 1; i++)
    {
        H[2*(j + i * (1 + dim))+0] = 1.0;
    }

#ifdef ZGEEV
    int N = dim;
    int LDH = dim, LDZ = dim;
    int INFO = 0;

    int lwork;
    char c = 'N';
    ComplexDouble wkopt;
    double* work;
    double *rwork = new double[2 * dim];

    lwork = -1;
    zgeev_(&c, &c, &N, H, &LDH, W, Z, &LDZ, Z, &LDZ, (double*)&wkopt, &lwork, rwork, &INFO );
    lwork = (int)wkopt.r;
    work = new double[2 * lwork];

    zgeev_(&c, &c, &N, H, &LDH, W, Z, &LDZ, Z, &LDZ, work, &lwork, rwork, &INFO );

#else
    int WANTT = 0, WANTZ = 0;
    int N = dim;
    int ILO = 1, ILOZ = 1;
    int IHI = dim, IHIZ = dim;
    int LDH = dim, LDZ = dim;
    int INFO = 0;

    zlahqr_(&WANTT, &WANTZ, &N, &ILO, &IHI, H, &LDH, W, &ILOZ, &IHIZ, Z, &LDZ, &INFO);

#endif

    for (int i = 0; i < dim; i++)
    {
        rE[i * n_matrices] = W[2*i+0];
        iE[i * n_matrices] = W[2*i+1];
    }

    delete [] H;
    delete [] W;
}
#endif

cuda void zlahqr_(double* rC, double* iC, double* rE, double* iE, double* rV, double* iV, int dim, int block_cnt, int thread_cnt)
{
    int n_matrices = thread_cnt;

    for (int i = 0; i < dim * dim - dim; i++)
    {
        rV[i * n_matrices] = 0.0;
        iV[i * n_matrices] = 0.0;
    }

    double cr = rC[dim * n_matrices];
    double ci = iC[dim * n_matrices];

    int j = (dim * dim - dim) * n_matrices;
    for (int i = 0; i < dim; i++)
    {
        double a = rC[i * n_matrices];
        double b = iC[i * n_matrices];
        cdiv(a, b, cr, ci);
        rV[j + i * n_matrices] = -a;
        iV[j + i * n_matrices] = -b;
    }

    j = n_matrices;
    for (int i = 0; i < dim - 1; i++)
    {
        rV[j + i * (n_matrices + dim * n_matrices)] = 1.0;
    }

    int K, L, M, I1, I2;
    int ILO = 1, IHI = dim;
    int i1, i2, i3, i4, i5, i6;
    double tr, ti;
    double v0r = 0.0, v0i = 0.0, v1r = 0.0, v1i = 0.0;
    double d, wr, sx;
    double wr1, wi1, wr2, wi2, wr3, wr4, wi4, wr5, wi5, wr6, zr, zi;
    double tst, d2, d3, aa, ab, ba, bb, s;
    double h1r, h1i, h2r, h2i;

    int off = n_matrices + dim * n_matrices, off2 = n_matrices;

    double safemin = DBL_MIN;
    double ulp = DBL_EPSILON;
    double smlnum = safemin * (((double) (IHI - ILO + 1)) / ulp);

    for (int I = IHI; I >= ILO; --I)
    {
        L = ILO;
        for (int ITS = 0; ITS <= 30; ++ITS)
        {
            for (K = I; K >= L + 1; --K)
            {
                i4 = (K + (K - 1) * dim) * n_matrices - off;
                wr4 = rV[i4];
                wi4 = iV[i4];

                if (cabs1(wr4, wi4) <= smlnum)
                {
                    break;
                }

                i1 = (K - 1 + (K - 1) * dim) * n_matrices - off;
                i2 = (K + K * dim) * n_matrices - off;
                wr1 = rV[i1];
                wi1 = iV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                tst = cabs1(wr1, wi1) + cabs1(wr2, wi2);

                if (tst == 0.0)
                {
                    if (K - 2 >= ILO)
                    {
                        i3 = (K - 1 + (K - 2) * dim) * n_matrices - off;
                        wr3 = rV[i3];
                        tst += dabs(wr3);
                    }
                    if (K + 1 <= IHI)
                    {
                        i6 = (K + 1 + K * dim) * n_matrices - off;
                        wr6 = iV[i6];
                        tst += dabs(wr6);
                    }
                }

                if (dabs(wr4) <= ulp * tst)
                {
                    i5 = (K - 1 + K * dim) * n_matrices - off;
                    wr5 = rV[i5];
                    wi5 = iV[i5];
                    d2 = cabs1(wr4, wi4);
                    d3 = cabs1(wr5, wi5);
                    ab = fmax(d2, d3);
                    ba = fmin(d2, d3);
                    zr = wr1 - wr2;
                    zi = wi1 - wi2;
                    d2 = cabs1(wr2, wi2);
                    d3 = cabs1(zr, zi);
                    aa = fmax(d2, d3);
                    bb = fmin(d2, d3);
                    s = aa + ab;

                    if (ba * (ab / s) <= fmax(smlnum, ulp * (bb * (aa / s))))
                    {
                        break;
                    }
                }
            }

            L = K;

            if (L > ILO)
            {
                i1 = (L + (L - 1) * dim) * n_matrices - off;
                rV[i1] = 0.0;
                iV[i1] = 0.0;
            }

            if (L >= I)
            {
                break;
            }

            I1 = L;
            I2 = I;

            if (ITS == 10)
            {
                i1 = (L + 1 + L * dim) * n_matrices - off;
                i2 = i1 - n_matrices;
                wr = rV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                d = dabs(wr) * 0.75;
                tr = d + wr2;
                ti = wi2;
            }
            else if (ITS == 20)
            {
                i1 = (I + (I - 1) * dim) * n_matrices - off;
                i2 = i1 + dim * n_matrices;
                wr = rV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                d = dabs(wr) * 0.75;
                tr = d + wr2;
                ti = wi2;
            }
            else
            {
                double ur, ui, vr, vi, h3r, h3i, xr, xi, yr, yi, z1r, z1i, z2r, z2i;

                i1 = (I + I * dim) * n_matrices - off;
                i2 = (I - 1 + I * dim) * n_matrices - off;
                i3 = (I + (I - 1) * dim) * n_matrices - off;
                i4 = ((I - 1) + (I - 1) * dim) * n_matrices - off;

                tr = rV[i1];
                ti = iV[i1];
                h2r = rV[i2];
                h2i = iV[i2];
                h3r = rV[i3];
                h3i = iV[i3];

                csqrt(h2r, h2i);
                csqrt(h3r, h3i);
                cmul(h2r, h2i, h3r, h3i);
                ur = h2r;
                ui = h2i;
                s = cabs1(ur, ui);
                if (s != 0.0)
                {
                    xr = rV[i4];
                    xi = iV[i4];
                    csub(xr, xi, tr, ti);
                    cmul(xr, xi, 0.5);
                    sx = cabs1(xr, xi);
                    s = fmax(s, sx);
                    z1r = xr;
                    z1i = xi;
                    cdiv(z1r, z1i, s);
                    z2r = ur;
                    z2i = ui;
                    cdiv(z2r, z2i, s);
                    cmul(z1r, z1i, z1r, z1i);
                    cmul(z2r, z2i, z2r, z2i);
                    cadd(z1r, z1i, z2r, z2i);
                    csqrt(z1r, z1i);
                    cmul(z1r, z1i, s);
                    yr = z1r;
                    yi = z1i;

                    if (sx > 0.0)
                    {
                        zr = xr / sx;
                        zi = xi / sx;

                        if (zr * yr + zi * yi < 0.0)
                        {
                            yr = -yr;
                            yi = -yi;
                        }
                    }
                    vr = ur;
                    vi = ui;
                    cadd(xr, xi, yr, yi);

#ifdef MKL
                    cdivex(ur, ui, xr, xi, ur, ui);
#else
                    cdiv(ur, ui, xr, xi);
#endif

                    cmul(ur, ui, vr, vi);
                    csub(tr, ti, ur, ui);
                }
            }

            double d1r;
            M = L;

            i1 = (L + L * dim) * n_matrices - off;
            i3 = (L + 1 + L * dim) * n_matrices - off;

            h1r = rV[i1];
            h1i = iV[i1];
            d1r = rV[i3];

            zr = h1r - tr;
            zi = h1i - ti;
            s = cabs1(zr, zi) + dabs(d1r);
            cdiv(zr, zi, s);
            d1r /= s;
            v0r = zr;
            v0i = zi;
            v1r = d1r;
            v1i = 0.0;

            single_shift_qr(rV, iV, I, L, M, I1, I2, v0r, v0i, v1r, v1i, dim, n_matrices);

        }
    }

    for (int I = IHI; I >= ILO; --I)
    {
        i1 = (I + I * dim) * n_matrices - off;
        i2 = I * n_matrices - off2;
        zr = rV[i1];
        zi = iV[i1];
        rE[i2] = zr;
        iE[i2] = zi;
    }
}

extern "C" double dlamch_(char *cmach);

global void zlahqr(double* rC, double* iC, double* rE, double* iE, double* rV, double* iV, int dim, int dim1, int dim2, int block_cnt, int thread_cnt)
{

#ifndef CUDA

#ifdef MIC
    int n = block_cnt * thread_cnt * dim, n1 = block_cnt * thread_cnt * dim1, n2 = block_cnt * thread_cnt * dim2;
#pragma omp target map(alloc: rV[0:n2], iV[0:n2]) map(to:rC[0:n1], iC[0:n1]) map(from: rE[0:n], iE[0:n])
#endif

#ifdef OPENMP

#ifdef LAPACK
    char s;
    dlamch_(&s);
#endif
    
#pragma omp parallel for
#endif

#ifdef OPENACC
    int n = block_cnt * thread_cnt * dim, n1 = block_cnt * thread_cnt * dim1, n2 = block_cnt * thread_cnt * dim2;
#pragma acc parallel loop create(rV[0:n2],iV[0:n2]) copyin(rC[0:n1],iC[0:n1]) copyout(rE[0:n],iE[0:n])
#endif

    for (int block_idx = 0; block_idx < block_cnt; block_idx++)
    {
        for (int thread_idx = 0; thread_idx < thread_cnt; thread_idx++)
        {

#else

            int block_idx = blockIdx.x;
            int thread_idx = threadIdx.x;

#endif

            ZLAHQR(rC + thread_idx + block_idx * thread_cnt * dim1, iC + thread_idx + block_idx * thread_cnt * dim1,
                    rE + thread_idx + block_idx * thread_cnt * dim, iE + thread_idx + block_idx * thread_cnt * dim,
                    rV + thread_idx + block_idx * thread_cnt * dim2, iV + thread_idx + block_idx * thread_cnt * dim2, dim, block_cnt, thread_cnt);

#ifndef CUDA

        }
    }

#endif
}

#endif
