using DynamicalSystems

# 1. Определение системы уравнений
function pendulums!(du, u, p, t)
    γ, λ, k = p
    φ1, v1, φ2, v2 = u

    du[1] = v1
    du[2] = γ - λ * v1 - sin(φ1) + k * sin(φ2 - φ1)
    du[3] = v2
    du[4] = γ - λ * v2 - sin(φ2) + k * sin(φ1 - φ2)
    return nothing
end

# 2. Параметры
γ_val = 0.3946
λ_val = 0.2
k_val = 0.3841
p = [γ_val, λ_val, k_val]

# Начальные условия
u0 = [1.0, 1.0, 0.0, 0.0]

# 3. Инициализация системы
ds = CoupledODEs(pendulums!, u0, p)

println("--- Запуск расчета спектра показателей Ляпунова ---")
println("Параметры: gamma = $γ_val, lambda = $λ_val, k = $k_val")

# 4. Расчет спектра
# N = 10000 — количество шагов интегрирования
# Δt = 0.01 — шаг по времени
# Ttr = 1000 — отбрасываем переходный процесс
spectrum = lyapunovspectrum(ds, 1000000; Δt = 0.01, Ttr = 1000, show_progress = true)

# 5. Вывод результатов
println("\nРезультаты (спектр Ляпунова):")
for (i, λ) in enumerate(spectrum)
    println("L$i = ", λ)
end

# Проверка условий хаоса
max_l = spectrum[1]
if max_l > 0.001
    println("\nВЫВОД: Обнаружен ДЕТЕРМИНИРОВАННЫЙ ХАОС (L1 > 0)")
else
    println("\nВЫВОД: Динамика РЕГУЛЯРНАЯ (L1 <= 0)")
end

# Проверка суммы (для диссипативной системы сумма должна быть < 0)
println("Сумма показателей: ", sum(spectrum))