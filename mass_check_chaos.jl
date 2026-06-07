using DynamicalSystems
using Printf

# 1. Определение системы
function pendulums!(du, u, p, t)
    γ, λ, k = p
    φ1, v1, φ2, v2 = u
    du[1] = v1
    du[2] = γ - λ * v1 - sin(φ1) + k * sin(φ2 - φ1)
    du[3] = v2
    du[4] = γ - λ * v2 - sin(φ2) + k * sin(φ1 - φ2)
    return nothing
end

# 2. Настройки
fixed_gamma = 0.99  # Берем из твоего конфига
u0 = [0.1, 0.5, 0.0, 0.0]
threshold = 0.001

input_file = "C:/Lobach4/tu/kneadings-master1/output/kneadings_pendulums1.txt"
output_file = "C:/Lobach4/tu/kneadings-master1/output/chaos_verification_report.txt"

found_chaos_points = String[]

println("--- МАССОВАЯ ПРОВЕРКА ПОКАЗАТЕЛЕЙ ЛЯПУНОВА ---")
println("Варьируем k и lambda. Фиксированная gamma = $fixed_gamma")

total_checked = 0

for line in eachline(input_file)
    # Ищем строки, где есть CHAOS
    if occursin("CHAOS", line)
        try
            # Ищем k и lambda (теперь ищем слово lambda!)
            m_k = match(r"k:\s*([-\d\.]+)", line)
            m_l = match(r"lambda:\s*([-\d\.]+)", line)

            if m_k !== nothing && m_l !== nothing
                global total_checked += 1
                k_val = parse(Float64, m_k.captures[1])
                λ_val = parse(Float64, m_l.captures[1])

                # Формируем вектор параметров [gamma, lambda, k]
                p = [fixed_gamma, λ_val, k_val]

                ds = CoupledODEs(pendulums!, u0, p)

                # Считаем подольше (5000), так как затухание маленькое
                spec = lyapunovspectrum(ds, 5000; Ttr = 1000)
                l1 = spec[1]

                if l1 > threshold
                    result_str = @sprintf("k: %.6f, lambda: %.6f => L1 = %.6f", k_val, λ_val, l1)
                    push!(found_chaos_points, result_str)
                    @printf("!!! НАЙДЕН ХАОС: %s\n", result_str)
                else
                    # Печатаем каждую 100-ю точку, чтобы не засорять консоль
                    if total_checked % 100 == 0
                        @printf("Проверено %d точек... Последняя: k=%.4f, l=%.4f (L1=%.6f)\n", total_checked, k_val, λ_val, l1)
                    end
                end
            end
        catch e
            # Пропускаем строки заголовка, где нет цифр
        end
    end
end

# Запись отчета
open(output_file, "w") do f
    write(f, "ОТЧЕТ О ВЕРИФИКАЦИИ ХАОСА (Сетка k-lambda)\n")
    write(f, "======================================\n")
    write(f, "Фиксированная gamma: $fixed_gamma\n")
    write(f, "Всего CHAOS-точек в файле: $total_checked\n\n")

    if isempty(found_chaos_points)
        write(f, "РЕЗУЛЬТАТ: Истинный хаос (L1 > 0) не обнаружен.\n")
    else
        write(f, "РЕЗУЛЬТАТ: Обнаружен хаос в $(length(found_chaos_points)) точках:\n")
        for p in found_chaos_points
            write(f, p * "\n")
        end
    end
end

println("\nГотово! Результаты в файле chaos_verification_report.txt")