import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY


frames = 200

t = sp.Symbol('t')

r = 1 + sp.sin(8*t)
phi = t + 0.5 * sp.sin(8*t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
v = (Vx ** 2 + Vy ** 2) ** 0.5 

Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
w = (Wx ** 2 + Wy ** 2) ** 0.5

Wtan = sp.diff(v, t) # получили модуль тангенциального ускорения в каждый момент времени
Wnor = (w ** 2 - Wtan ** 2) ** 0.5 # нашли нормальное ускорение как разность полного и тангенциального в квадратах

# ищем модуль радиуса кривизны 
curvatureRadius = v*v/Wnor

# находим координаты вектора тангенциального ускорения:
# нормируем вектор скорости и умножаем на величину тангенциального
WTanx = Vx / v * Wtan
WTany = Vy / v * Wtan

# N - единичный вектор, сонаправленный с нормальным ускорением
# Вычитая из координат полного ускорения координаты тангенциального ускорения,
# получаем координаты нормального ускорения
WNorX = Wx - WTanx
WNorY = Wy - WTany

WNor = (WNorX ** 2 + WNorY ** 2) ** 0.5

Nx = WNorX / WNor
Ny = WNorY / WNor

curvatureRadiusx = Nx * curvatureRadius
curvatureRadiusy = Ny * curvatureRadius

Time = np.linspace(0, 2, frames) # 10 секунд времени разделенные на frames частей

X_dot = np.zeros_like(Time)
Y_dot = np.zeros_like(Time)
VX = np.zeros_like(Time)
VY = np.zeros_like(Time)
AX = np.zeros_like(Time)
AY = np.zeros_like(Time)
RadiusVectorX = np.zeros_like(Time)
RadiusVectorY = np.zeros_like(Time)
CurvatureRadiusX = np.zeros_like(Time)
CurvatureRadiusY = np.zeros_like(Time)

# считаем все значения на нашем промежутке времени
for i in np.arange(len(Time)):
    # точка
    X_dot[i] = sp.Subs(x, t, Time[i]) # в функицию х посдтавляет вместо t значение T[i]
    Y_dot[i] = sp.Subs(y, t, Time[i])
    
	# скорость
    VX[i] = sp.Subs(Vx, t, Time[i])
    VY[i] = sp.Subs(Vy, t, Time[i])
    
	# ускорение
    AX[i] = sp.Subs(Wx, t, Time[i])
    AY[i] = sp.Subs(Wy, t, Time[i])
    
	# радиус вектор
    RadiusVectorX[i] = sp.Subs(x, t, Time[i])
    RadiusVectorY[i] = sp.Subs(y, t, Time[i])
    
	# радиус кривизны
    CurvatureRadiusX[i] = sp.Subs(curvatureRadiusx, t, Time[i])
    CurvatureRadiusY[i] = sp.Subs(curvatureRadiusy, t, Time[i])


# задаем условия графика
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[int(X_dot.min()) - 1, int(X_dot.max()) + 1], ylim=[int(Y_dot.min()) - 1, int(Y_dot.max()) + 1])

# рисуем сразу всю траекторию движения точки
ax1.plot(X_dot, Y_dot) 

# рисуем оси координат
ax1.plot([min(0, X_dot.min()), max(0, X_dot.max())], [0, 0], 'black')
ax1.plot([0, 0], [min(0, Y_dot.min()), max(0, Y_dot.max())], 'black')

# рисуем точку в начальный момент времени
P, = ax1.plot(X_dot[0], Y_dot[0], marker='o') 

# вектор скорости
VLine, = ax1.plot([X_dot[0], X_dot[0]+VX[0]], [Y_dot[0], Y_dot[0]+VY[0]], 'r')

# вектор ускорения
ALine, = ax1.plot([X_dot[0], X_dot[0]+AX[0]], [Y_dot[0], Y_dot[0]+AY[0]], 'g')

# радиус вектор
RadiusVector, = ax1.plot([0, X_dot[0]], [0, Y_dot[0]], 'c')

# радиус кривизны
CurvatureRadiusVector, = ax1.plot([X_dot[0], X_dot[0] + CurvatureRadiusX[0]], [Y_dot[0], Y_dot[0] + CurvatureRadiusY[0]], 'b')


# задаем красивую стрелочку
arrowMult = 0.5
ArrowX = np.array([-0.2*arrowMult, 0, -0.2*arrowMult])
ArrowY = np.array([0.1*arrowMult, 0, -0.1*arrowMult])

# стрелочка для вектора скорости
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX+X_dot[0]+VX[0], RArrowY+Y_dot[0]+VY[0], 'r')

# стрелочка для вектора ускорения
AArrowX, AArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))
AArrow, = ax1.plot(AArrowX+X_dot[0]+AX[0], AArrowY+Y_dot[0]+AY[0], 'g')

# стрелочка для радиус вектора
RadiusVectorArrowX, RadiusVectorArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
RadiusVectorArrow, = ax1.plot(RadiusVectorArrowX+X_dot[0], RadiusVectorArrowY+Y_dot[0], 'c')

# стрелочка для радиуса кривизны
CurvatureRadiusVectorArrowX, CurvatureRadiusVectorArrowY = Rot2D(
	ArrowX, ArrowY, math.atan2(CurvatureRadiusY[0], CurvatureRadiusX[0])
	)
CurvatureRadiusVectorArrow, = ax1.plot(
    CurvatureRadiusVectorArrowX+CurvatureRadiusX[0], CurvatureRadiusVectorArrowY+CurvatureRadiusY[0], 'b'
    )


def anima(i):
	P.set_data([X_dot[i]], [Y_dot[i]]) # изменяем положение точки, меняем координаты на соответствующие времени

	# вектор скорости
	VLine.set_data([X_dot[i], X_dot[i]+VX[i]], [Y_dot[i], Y_dot[i]+VY[i]])
     
	# вектор ускорения
	ALine.set_data([X_dot[i], X_dot[i]+AX[i]], [Y_dot[i], Y_dot[i]+AY[i]])

	# радиус вектор
	RadiusVector.set_data([0, X_dot[i]], [0, Y_dot[i]])
     
	# радиус кривизны
	CurvatureRadiusVector.set_data([X_dot[i], X_dot[i] + CurvatureRadiusX[i]], [Y_dot[i], Y_dot[i] + CurvatureRadiusY[i]])
     
	# стрелочка для вектора скорости
	RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
	VArrow.set_data(RArrowX+X_dot[i]+VX[i], RArrowY+Y_dot[i]+VY[i])
    
	# стрелочка для вектора ускорения
	AArrowX, AArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
	AArrow.set_data(AArrowX+X_dot[i]+AX[i], AArrowY+Y_dot[i]+AY[i])
     
	# стрелочка для радиус вектора
	RadiusVectorArrowX, RadiusVectorArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y_dot[i], X_dot[i]))
	RadiusVectorArrow.set_data(RadiusVectorArrowX+X_dot[i], RadiusVectorArrowY+Y_dot[i])
     
	# стрелочка для радиуса кривизны
	CurvatureRadiusVectorArrowX, CurvatureRadiusVectorArrowY = Rot2D(
        ArrowX, ArrowY, math.atan2(CurvatureRadiusY[i], CurvatureRadiusX[i])
    	)
	CurvatureRadiusVectorArrow.set_data(
        CurvatureRadiusVectorArrowX+X_dot[i] + CurvatureRadiusX[i], CurvatureRadiusVectorArrowY+Y_dot[i] + CurvatureRadiusY[i]
    	)


anim = FuncAnimation(fig, anima, frames=frames, interval=50, repeat=False)
plt.show()
