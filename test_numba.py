from numba import jit
import numpy as np
import time

@jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = np.random.random()
        y = np.random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

print("Compiling...")
start = time.time()
monte_carlo_pi(100)
print(f"Compilation time: {time.time() - start:.4f}s")

print("Running...")
start = time.time()
res = monte_carlo_pi(10000000)
print(f"Execution time: {time.time() - start:.4f}s")
print(f"Result: {res}")
