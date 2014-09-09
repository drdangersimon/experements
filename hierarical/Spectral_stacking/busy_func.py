import numpy as nu
from scipy.special import erf




'''B(x) = (a/4) * [erf(b1 (w + x - xe)) + 1] * [erf(b2 (w - x + xe)) + 1]
      * (c |x - xp|^n + 1)


     where erf() is the error function. There are up to eight free parameters (some of which can be merged or fixed, if desired): amplitude (a), slope of the line flanks (b1, b2), profile width (w), position of the overall profile (xe) and trough (xp), relative depth of the trough (c), and shape/order of the polynomial trough (n). '''


def Busy(x, amp, flank1, flank2, width, xprof, xtrough, deptrough, n):
    '''Busy function for double horn profiles of HI in galaxies.
    See Westmeier et. al. 2013 for more info

    Inputs are:
    frequency (x)
    amplitude (amp)
    slope of the line flanks (flank1, flank2)
    profile width (width)
    position of the overall profile (xprof) and trough (xtrough)
    relative depth of the trough (deptrough)
    shape/order of the polynomial trough (n)

    output same shape as x'''

    
    
    B = ((amp / 4.) * (erf(flank1 * (width + x - xprof)) + 1) *
        (erf(flank2 * (width - x + xprof)) + 1) *
        (deptrough * (nu.abs(x - xtrough)**n + 1)))

    return B
