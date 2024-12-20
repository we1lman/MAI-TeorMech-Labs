import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


frames = 200
t = sp.Symbol('t')

# Генерируем временные значения
Time = np.linspace(0, 2, frames)  # 2 секунды, можно изменить по заданию

# Определяем символьные выражения
r = 1 + sp.sin(8 * t)
phi = t + 0.5 * sp.sin(8 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
v = (Vx ** 2 + Vy ** 2) ** 0.5

Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
w = (Wx ** 2 + Wy ** 2) ** 0.5

Wtan = sp.diff(v, t)  # тангенциальное ускорение
Wnor = sp.sqrt(w ** 2 - Wtan ** 2)  # нормальное ускорение

# Радиус кривизны
curvatureRadius = v * v / Wnor

# Векторы тангенциального и нормального ускорения
WTanx = Vx / v * Wtan
WTany = Vy / v * Wtan
WNorX = Wx - WTanx
WNorY = Wy - WTany
WNor = sp.sqrt(WNorX ** 2 + WNorY ** 2)
Nx = WNorX / WNor
Ny = WNorY / WNor

curvatureRadiusx = Nx * curvatureRadius
curvatureRadiusy = Ny * curvatureRadius

# Создаём числовые функции с помощью lambdify
x_func = sp.lambdify(t, x, 'numpy')
y_func = sp.lambdify(t, y, 'numpy')
Vx_func = sp.lambdify(t, Vx, 'numpy')
Vy_func = sp.lambdify(t, Vy, 'numpy')
Wx_func = sp.lambdify(t, Wx, 'numpy')
Wy_func = sp.lambdify(t, Wy, 'numpy')
curvatureRadiusx_func = sp.lambdify(t, curvatureRadiusx, 'numpy')
curvatureRadiusy_func = sp.lambdify(t, curvatureRadiusy, 'numpy')

# Вычисляем численные значения
X_dot = x_func(Time)
Y_dot = y_func(Time)
VX = Vx_func(Time)
VY = Vy_func(Time)
AX = Wx_func(Time)
AY = Wy_func(Time)
RadiusVectorX = X_dot
RadiusVectorY = Y_dot
CurvatureRadiusX = curvatureRadiusx_func(Time)
CurvatureRadiusY = curvatureRadiusy_func(Time)

# Задаём условия графика
fig, ax1 = plt.subplots()
ax1.axis('equal')
ax1.set(xlim=[X_dot.min() - 1, X_dot.max() + 1], ylim=[Y_dot.min() - 1, Y_dot.max() + 1])

# Рисуем траекторию
ax1.plot(X_dot, Y_dot, label='Траектория')

# Рисуем оси координат
ax1.axhline(0, color='black')
ax1.axvline(0, color='black')

# Инициализируем элементы анимации
P, = ax1.plot([], [], marker='o', color='k', label='Точка')
VLine, = ax1.plot([], [], 'r-', label='Скорость')
ALine, = ax1.plot([], [], 'g-', label='Ускорение')
RadiusVector, = ax1.plot([], [], 'c-', label='Радиус-вектор')
CurvatureRadiusVector, = ax1.plot([], [], 'b-', label='Радиус кривизны')

# Шаблон стрелок для наконечников
ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])
VArrow, = ax1.plot([], [], 'r')  # Стрелка для скорости
AArrow, = ax1.plot([], [], 'g')  # Стрелка для ускорения
RArrow, = ax1.plot([], [], 'c')  # Стрелка для радиус-вектора
CRArrow, = ax1.plot([], [], 'b')  # Стрелка для радиуса кривизны


# Функция анимации
def anima(i):
    # Обновляем положение точки
    P.set_data([X_dot[i]], [Y_dot[i]])

    # Вектор скорости
    VLine.set_data([X_dot[i], X_dot[i] + VX[i]], [Y_dot[i], Y_dot[i] + VY[i]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX + X_dot[i] + VX[i], RArrowY + Y_dot[i] + VY[i])

    # Вектор ускорения
    ALine.set_data([X_dot[i], X_dot[i] + AX[i]], [Y_dot[i], Y_dot[i] + AY[i]])
    RWArrowX, RWArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AArrow.set_data(RWArrowX + X_dot[i] + AX[i], RWArrowY + Y_dot[i] + AY[i])

    # Радиус-вектор
    RadiusVector.set_data([0, X_dot[i]], [0, Y_dot[i]])
    rArrowX, rArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y_dot[i], X_dot[i]))
    RArrow.set_data(rArrowX + X_dot[i], rArrowY + Y_dot[i])

    # Радиус кривизны
    CurvatureRadiusVector.set_data(
        [X_dot[i], X_dot[i] + CurvatureRadiusX[i]], [Y_dot[i], Y_dot[i] + CurvatureRadiusY[i]]
    )
    CRArrowX, CRArrowY = Rot2D(ArrowX, ArrowY, math.atan2(CurvatureRadiusY[i], CurvatureRadiusX[i]))
    CRArrow.set_data(CRArrowX + X_dot[i] + CurvatureRadiusX[i], CRArrowY + Y_dot[i] + CurvatureRadiusY[i])

    return P, VLine, VArrow, ALine, AArrow, RadiusVector, RArrow, CurvatureRadiusVector, CRArrow


# Добавляем легенду
ax1.legend()

# Создаём анимацию
anim = FuncAnimation(fig, anima, frames=frames, interval=50, blit=True, repeat=False)
plt.show()
