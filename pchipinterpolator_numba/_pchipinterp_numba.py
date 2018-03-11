# -*- coding: utf-8 -*-
'''
Created on 11 mars 2018

@author: Xavier
'''
from scipy.interpolate import PchipInterpolator
import numpy as np
import numba as nb

__all__ = ['PchipInterpolatorNumba']

class PchipInterpolatorNumba(PchipInterpolator):
    def _evaluate(self, x, nu, extrapolate, out):
        evaluate_bernstein_float(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x, x, nu, bool(extrapolate), out)
        
@nb.njit(cache=True)
def find_interval_ascending(x, nx, xval, prev_interval=0, extrapolate=1):
    """
    Find an interval such that x[interval] <= xval < x[interval+1]. Assuming
    that x is sorted in the ascending order.
    If xval < x[0], then interval = 0, if xval > x[-1] then interval = n - 2.
    Parameters
    ----------
    x : array of double, shape (m,)
        Piecewise polynomial breakpoints sorted in ascending order.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.
    extrapolate : bint, optional
        Whether to return the last of the first interval if the
        point is out-of-bounds.
    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.
    """
    a = x[0]
    b = x[nx-1]

    interval = prev_interval
    if interval < 0 or interval >= nx:
        interval = 0

    if not (a <= xval <= b):
        # Out-of-bounds (or nan)
        if xval < a and extrapolate:
            # below
            interval = 0
        elif xval > b and extrapolate:
            # above
            interval = nx - 2
        else:
            # nan or no extrapolation
            interval = -1
    elif xval == b:
        # Make the interval closed from the right
        interval = nx - 2
    else:
        # Find the interval the coordinate is in
        # (binary search with locality)
        if xval >= x[interval]:
            low = interval
            high = nx - 2
        else:
            low = 0
            high = interval

        if xval < x[low+1]:
            high = low

        while low < high:
            mid = (high + low)//2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 1
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid
                break

        interval = low

    return interval

        
@nb.njit(cache=True)
def find_interval_descending(x, nx, xval, prev_interval=0, extrapolate=1): 
    """
    Find an interval such that x[interval + 1] < xval <= x[interval], assuming
    that x are sorted in the descending order.
    If xval > x[0], then interval = 0, if xval < x[-1] then interval = n - 2.
    Parameters
    ----------
    x : array of double, shape (m,)
        Piecewise polynomial breakpoints sorted in descending order.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.
    extrapolate : bint, optional
        Whether to return the last of the first interval if the
        point is out-of-bounds.
    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.
    """
    # Note that now a > b.
    a = x[0]
    b = x[nx-1]

    interval = prev_interval
    if interval < 0 or interval >= nx:
        interval = 0

    if not (b <= xval <= a):
        # Out-of-bounds or NaN.
        if xval > a and extrapolate:
            # Above a.
            interval = 0
        elif xval < b and extrapolate:
            # Below b.
            interval = nx - 2
        else:
            # No extrapolation.
            interval = -1
    elif xval == b:
        # Make the interval closed from the left.
        interval = nx - 2
    else:
        # Apply the binary search in a general case. Note that low and high
        # are used in terms of interval number, not in terms of abscissas.
        # The conversion from find_interval_ascending is simply to change
        # < to > and >= to <= in comparison with xval.
        if xval <= x[interval]:
            low = interval
            high = nx - 2
        else:
            low = 0
            high = interval

        if xval > x[low + 1]:
            high = low

        while low < high:
            mid = (high + low) // 2
            if xval > x[mid]:
                # mid < high
                high = mid
            elif xval <= x[mid + 1]:
                low = mid + 1
            else:
                # x[mid] >= xval > x[mid+1]
                low = mid
                break

        interval = low

    return interval


#------------------------------------------------------------------------------
# Piecewise Bernstein basis polynomials
#------------------------------------------------------------------------------
@nb.njit(cache=True)
def evaluate_bpoly1(s, c, ci, cj):
    """
    Evaluate polynomial in the Bernstein basis in a single interval.
    A Bernstein polynomial is defined as
        .. math:: b_{j, k} = comb(k, j) x^{j} (1-x)^{k-j}
    with ``0 <= x <= 1``.
    Parameters
    ----------
    s : double
        Polynomial x-value
    c : double[:,:,:]
        Polynomial coefficients. c[:,ci,cj] will be used
    ci, cj : int
        Which of the coefs to use
    """
 
    k = c.shape[0] - 1  # polynomial order
    s1 = 1. - s
 
    # special-case lowest orders
    if k == 0:
        res = c[0, ci, cj]
    elif k == 1: 
        res = c[0, ci, cj] * s1 + c[1, ci, cj] * s
    elif k == 2:
        res = c[0, ci, cj] * s1*s1 + c[1, ci, cj] * 2.*s1*s + c[2, ci, cj] * s*s
    elif k == 3:
        res = (c[0, ci, cj] * s1*s1*s1 + c[1, ci, cj] * 3.*s1*s1*s +
               c[2, ci, cj] * 3.*s1*s*s + c[3, ci, cj] * s*s*s)
    else:
        # XX: replace with de Casteljau's algorithm if needs be
        res, comb = 0., 1.
        for j in range(k+1):
            res += comb * s**j * s1**(k-j) * c[j, ci, cj]
            comb *= 1. * (k-j) / (j+1.)
 
    return res
 
@nb.njit(cache=True)
def evaluate_bpoly1_deriv(s, c, ci, cj, nu, wrk):
    """
    Evaluate the derivative of a polynomial in the Bernstein basis 
    in a single interval.
    A Bernstein polynomial is defined as
        .. math:: b_{j, k} = comb(k, j) x^{j} (1-x)^{k-j}
    with ``0 <= x <= 1``.
    The algorithm is detailed in BPoly._construct_from_derivatives.
    Parameters
    ----------
    s : double
        Polynomial x-value
    c : double[:,:,:]
        Polynomial coefficients. c[:,ci,cj] will be used
    ci, cj : int
        Which of the coefs to use
    nu : int
        Order of the derivative to evaluate. Assumed strictly positive
        (no checks are made).
    wrk : double[:,:,::1]
        A work array, shape (c.shape[0]-nu, 1, 1).
    """
    k = c.shape[0] - 1  # polynomial order
 
    if nu == 0:
        res = evaluate_bpoly1(s, c, ci, cj)
    else:
        poch = 1.
        for a in range(nu):
            poch *= k - a
 
        term = 0.
        for a in range(k - nu + 1):
            term, comb = 0., 1.
            for j in range(nu+1):
                term += c[j+a, ci, cj] * (-1)**(j+nu) * comb
                comb *= 1. * (nu-j) / (j+1)
            wrk[a, 0, 0] = term * poch
        res = evaluate_bpoly1(s, wrk, 0, 0)
    return res
 
#
# Evaluation; only differs from _ppoly by evaluate_poly1 -> evaluate_bpoly1
#
@nb.njit(parallel=True)
def evaluate_bernstein_float(c, x, xp, nu, extrapolate, out):
    """
    Evaluate a piecewise polynomial in the Bernstein basis.
    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    nu : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : bint, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.
    """
    
    # check derivative order
    if nu < 0:
        raise NotImplementedError("Cannot do antiderivatives in the B-basis yet.")
 
    # shape checks
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[2]:
        raise ValueError("out and c have incompatible shapes")
    if c.shape[1] != x.shape[0] - 1:
        raise ValueError("x and c have incompatible shapes")
 
    if nu > 0:
        wrk = np.empty((c.shape[0]-nu, 1, 1), dtype=np.float_)
         
    
    
    ascending = x[x.shape[0] - 1] >= x[0]
 
    # Evaluate.
    n_iter = 0
    interval_arr = np.zeros(len(xp),dtype=np.int64)
    for ip in nb.prange(len(xp)):
        n_iter += 1
        xval = xp[ip]
 
        # Find correct interval
        if n_iter > 0:
            interval_ini = interval_arr[n_iter-1]
        else:
            interval_ini = 0
        if ascending:
            
            i = find_interval_ascending(x, x.shape[0], xval, interval_ini,
                                        extrapolate)
        else:
            i = find_interval_descending(x, x.shape[0], xval, interval_ini,
                                         extrapolate)
  
        if i < 0:
            # xval was nan etc
            for jp in range(c.shape[2]):
                out[ip, jp] = np.nan
            continue
        else:
            interval = i
            interval_arr[n_iter] = interval
  
        # Evaluate the local polynomial(s)
        ds = x[interval+1] - x[interval]
        ds_nu = ds**nu
        for jp in range(c.shape[2]):
            s = (xval - x[interval]) / ds
            if nu == 0:
                out[ip, jp] = evaluate_bpoly1(s, c, interval, jp)
            else:
                out[ip, jp] = evaluate_bpoly1_deriv(s, c, interval, jp,
                        nu, wrk) / ds_nu