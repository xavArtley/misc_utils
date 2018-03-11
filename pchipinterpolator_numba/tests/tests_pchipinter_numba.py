# -*- coding: utf-8 -*-
'''
Created on 10 mars 2018

@author: Xavier
'''
import sys
sys.path.append('..')

from scipy.interpolate import PchipInterpolator
from pchipinterpolator_numba import PchipInterpolatorNumba
import numpy as np
import timeit

x = np.sort(np.random.rand(100))
y = x**2-np.sign(x-0.5)
xs = np.linspace(x.min(),x.max(),10000000)
pchip_interp_sp = PchipInterpolator(x,y)
pchip_interp_nb = PchipInterpolatorNumba(x,y)

def test_interpolation():
    """
    Compare interpolation results between cython (original) and numba versions of PchipInterpolator
    """
    assert np.allclose(pchip_interp_sp(xs), pchip_interp_nb(xs))

def test_derivative_1():
    """
    Compare derivative results between cython (original) and numba versions of PchipInterpolator
    """
    assert np.allclose(pchip_interp_sp.derivative(1)(xs), pchip_interp_nb.derivative(1)(xs))
    
def test_derivative_2():
    """
    Compare derivative results between cython (original) and numba versions of PchipInterpolator
    """
    assert np.allclose(pchip_interp_sp.derivative(2)(xs), pchip_interp_nb.derivative(2)(xs))

def test_benchmark():
    """
    Compare speed between cython (original) and numba versions of PchipInterpolator
    """
    time_sp = timeit.timeit('pchip_interp_sp(xs)', globals=globals(), number=20)
    time_nb = timeit.timeit('pchip_interp_nb(xs)', globals=globals(), number=20)
    time_sp_der_1 = timeit.timeit('pchip_interp_sp.derivative(1)(xs)', globals=globals(), number=20)
    time_nb_der_1 = timeit.timeit('pchip_interp_nb.derivative(1)(xs)', globals=globals(), number=20)
    time_sp_der_2 = timeit.timeit('pchip_interp_sp.derivative(2)(xs)', globals=globals(), number=20)
    time_nb_der_2 = timeit.timeit('pchip_interp_nb.derivative(2)(xs)', globals=globals(), number=20)
    
    print('\n')
    print("Simple interpolation")
    print('Scipy Pchip: ', time_sp)
    print('Numba Pchip: ', time_nb)
    print("Derivative order 1")
    print('Scipy Pchip interpolation: ', time_sp_der_1)
    print('Numba Pchip interpolation: ', time_nb_der_1)
    print("Derivative order 2")
    print('Scipy Pchip interpolation: ', time_sp_der_2)
    print('Numba Pchip interpolation: ', time_nb_der_2)
    
def test_plot_results():
    """
    Plot results
    """
    import matplotlib.pyplot as plt
    plt.subplot(311)
    plt.plot(x,y,'ro')
    plt.plot(xs,pchip_interp_nb(xs),label='Interpolation')
    plt.subplot(312)
    plt.plot(xs,pchip_interp_nb.derivative(1)(xs),label='First derivative')
    plt.subplot(313)
    plt.plot(xs,pchip_interp_nb.derivative(2)(xs),label='Second derivative')
    
    plt.show()