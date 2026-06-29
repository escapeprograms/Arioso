import numpy as np

#evaluation
def FFT(x):
    N = len(x)

    #base case
    if N == 1:
        return x
    
    #Nth root of unity
    W = np.exp(-2j * np.pi / N)

    #break down
    x_even = x[0::2] # 0::2 selects every other element, starting from the 0th
    x_odd = x[1::2] # 1::2 selects every other element, starting from the 1st
    
    X_even = FFT(x_even)
    X_odd = FFT(x_odd)
    

    X = np.zeros(N, dtype=complex)
    
    for k in range(N // 2):
        X_odd_term = (W ** k) * X_odd[k]
        
        #k = 0 to N/2 - 1
        X[k] = X_even[k] + X_odd_term
        #k = N/2 to N - 1
        X[k + N // 2] = X_even[k] - X_odd_term
        
    return X


#interpolation: Inverse Fourier Transform
def IFFT(X):
    N = len(X)

    if N == 1:
        return X

    W = np.exp(2j * np.pi / N)

    X_even = X[0::2]
    X_odd = X[1::2]

    x_even = IFFT(X_even)
    x_odd = IFFT(X_odd)

    x = np.zeros(N, dtype=complex)

    for k in range(N // 2):
        x_odd_term = (W ** k) * x_odd[k]

        x[k] = (x_even[k] + x_odd_term) / 2
        x[k + N // 2] = (x_even[k] - x_odd_term) / 2

    return x
