import numpy as np
from numpy.fft import fft, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import hilbert

t = np.linspace( 0, 1, 1000, endpoint = False )
x = 10 + 10 * np.cos( 2 * np.pi * 100 * t ) + 2 * np.cos( 2 * np.pi * 200 * t )

a = fft(x)
plt.stem(a, markerfmt='o')
plt.xlabel('x')
plt.ylabel('Magnitude')
plt.show()
b = fft(hilbert(x))
plt.stem(b, markerfmt='o')
plt.xlabel('x')
plt.ylabel('Magnitude')
plt.show()

f = fftshift( fftfreq( 1000, 0.001 ) )
X = fftshift( fft( x ) )
Xm = abs( X )/len(X)

plt.plot( f, Xm )
plt.xlabel( 'f (Hz)' )
plt.ylabel( 'Magnitude' )

plt.show( )