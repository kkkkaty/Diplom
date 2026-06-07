import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# параметры
gamma = 0.994
lam = 0.2
k = 0.937



def system(t, y):
    fi1, v1, fi2, v2 = y

    dfi1 = v1
    dv1 = gamma - lam * v1 - np.sin(fi1) + k * np.sin(fi2 - fi1)

    dfi2 = v2
    dv2 = gamma - lam * v2 - np.sin(fi2) + k * np.sin(fi1 - fi2)

    return [dfi1, dv1, dfi2, dv2]


# начальное условие
y0 = [0.0, 0.0, 0.0, 0.0]

# время интегрирования
T = 8000

t_eval = np.linspace(0, T, 100000)

sol = solve_ivp(
    system,
    [0, T],
    y0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-9
)

t = sol.t
fi1 = sol.y[0]
fi2 = sol.y[2]

plt.figure(figsize=(12,6))

plt.plot(t, fi1, label=r'$\varphi_1(t)$')
plt.plot(t, fi2, label=r'$\varphi_2(t)$')

plt.xlabel('t')
plt.ylabel(r'$\varphi$')
plt.legend()
plt.grid()
print("fi1(T) =", fi1[-1])
print("fi2(T) =", fi2[-1])

print("N1 =", (fi1[-1] - fi1[0])/(2*np.pi))
print("N2 =", (fi2[-1] - fi2[0])/(2*np.pi))
plt.show()