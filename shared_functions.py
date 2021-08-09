"""
Documentation at https://github.com/dmacneill/AxisID. 
"""

import numpy as np

modulus = 30
k = int(360/modulus)

def angletovec(angle):
    
    """Calculates vector encoding of an angle
    """
    return np.column_stack((np.cos(k*angle*np.pi/180), np.sin(k*angle*np.pi/180)))

def vectoangle(vec):
    
    """Calculates angle from the vector encoding
    """
    slice_real = [slice(None)]*len(vec.shape)
    slice_imag = [slice(None)]*len(vec.shape)
    slice_real[-1] = 0 
    slice_imag[-1] = 1
    
    return ((1/k)*(180/np.pi)*np.imag(np.log(vec[tuple(slice_real)]+(1j)*vec[tuple(slice_imag)])))%modulus

def circular_difference(angle_1, angle_2):
    
    """Returns a signed version of the distance between two points at angles 
    angle_2 and angle_1 on a circle of perimeter modulus. The sign convention 
    is chosen to give a positive sign if the shortest arc from angle_1 to angle_2
    is counter-clockwise and negative otherwise. 
    """
    angle_1 = angle_1%modulus
    angle_2 = angle_2%modulus
    
    return (angle_2-angle_1-modulus/2)%modulus-modulus/2