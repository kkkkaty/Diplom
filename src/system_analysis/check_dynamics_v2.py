import numpy as np
import matplotlib.pyplot as plt
from src.system_analysis.get_inits import build_separatrix_init_for_point, rk4_step

# --- НАСТРОЙКИ ТОЧКИ ---
k_val = 1.5
gamma_val = 0.5
lam = 0.2
dt = 0.01
n_steps = 3100
skip_steps = 1000

def get_trajectory():
    eq, y0, _, _ = build_separatrix_init_for_point(
        gamma=gamma_val, lam=lam, k=k_val,
        saddle_focus_rule="phi1_lt_phi2",
        branch_rule="phi2_above_eq"
    )
    traj = np.zeros((n_steps, 4))
    y = y0
    for i in range(n_steps):
        y = rk4_step(y, dt, gamma_val, lam, k_val)
        traj[i] = y
    return traj

trajectory = get_trajectory()
steady_state = trajectory[skip_steps:]

# Используем сырые значения углов без свёртки
phi1 = steady_state[:, 0]
phi2 = steady_state[:, 2]
v1 = steady_state[:, 1]
v2 = steady_state[:, 3]

# --- ВИЗУАЛИЗАЦИЯ 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
fig.suptitle(f"k={k_val}, gamma={gamma_val}, lambda={lam}", fontsize=16)

# Ограничим количество отрисовываемых точек для четкости линий
plot_lim = 30000

# 1. Сверху-слева: Фазовый портрет 1
axes[0, 0].plot(phi1[:plot_lim], v1[:plot_lim], color='black', lw=0.4, alpha=0.6)
axes[0, 0].set_title(r"$\phi_1$ vs $v_1$")
axes[0, 0].set_xlabel(r"$\phi_1$")
axes[0, 0].set_ylabel(r"$v_1$")

# 2. Сверху-справа: Фазовый портрет 2
axes[0, 1].plot(phi2[:plot_lim], v2[:plot_lim], color='red', lw=0.4, alpha=0.6)
axes[0, 1].set_title(r"$\phi_2$ vs $v_2$")
axes[0, 1].set_xlabel(r"$\phi_2$")
axes[0, 1].set_ylabel(r"$v_2$")

# 3. Снизу-слева: Конфигурационная проекция
axes[1, 0].plot(phi1[:plot_lim], phi2[:plot_lim], color='blue', lw=0.3, alpha=0.5)
axes[1, 0].set_title(r"$\phi_1$ vs $\phi_2$")
axes[1, 0].set_xlabel(r"$\phi_1$")
axes[1, 0].set_ylabel(r"$\phi_2$")

# 4. Снизу-справа: Скоростная проекция
axes[1, 1].plot(v1[:plot_lim], v2[:plot_lim], color='darkgreen', lw=0.4, alpha=0.6)
axes[1, 1].set_title(r"$v_1$ vs $v_2$")
axes[1, 1].set_xlabel(r"$v_1$")
axes[1, 1].set_ylabel(r"$v_2$")

for ax in axes.flat:
    ax.grid(True, alpha=0.2)

# Автомасштабирование осей включено по умолчанию,
# поэтому графики корректно отобразят полный диапазон углов
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()