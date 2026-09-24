import numpy as np
import matplotlib.pyplot as plt

A = np.array([[-1.0, 1.0],
              [-4.0, 1.0]]) # mesmo procedimento do 1a 
w = np.sqrt(3.0)


def campo(x, y):
    return A[0, 0] * x + A[0, 1] * y, A[1, 0] * x + A[1, 1] * y


def H(x, y):
    return 4 * x**2 - 2 * x * y + y**2


def solucao(x0, y0, t):
    xd0 = -x0 + y0
    x = x0 * np.cos(w * t) + (xd0 / w) * np.sin(w * t)
    xd = -x0 * w * np.sin(w * t) + xd0 * np.cos(w * t)
    y = xd + x
    return x, y

lam = np.linalg.eigvals(A)
print("Autovalores:", lam)
print("tr A =", np.trace(A), "| det A =", round(np.linalg.det(A), 12))

t = np.linspace(0, 2 * np.pi / w, 2001)
x, y = solucao(0.5, 0.3, t)
print("Variacao relativa de H ao longo de 1 periodo:",
      np.ptp(H(x, y)) / H(0.5, 0.3))
print("Fechamento da orbita |r(T) - r(0)|:",
      np.hypot(x[-1] - x[0], y[-1] - y[0]))

L = 2.0
fig, ax = plt.subplots(figsize=(7, 7))

xg, yg = np.meshgrid(np.linspace(-L, L, 300), np.linspace(-L, L, 300))
u, v = campo(xg, yg)
ax.streamplot(xg, yg, u, v, color="0.75", density=1.2,
              linewidth=0.6, arrowsize=0.8)

niveis = [0.05, 0.2, 0.45, 0.8, 1.25, 1.8]
cs = ax.contour(xg, yg, H(xg, yg), levels=niveis,
                colors="tab:orange", linewidths=1.5)

for h in niveis:
    x0 = np.sqrt(h / 4)
    dx, dy = campo(x0, 0.0)
    n = np.hypot(dx, dy)
    ax.annotate("", xy=(x0 + 0.08 * dx / n, 0.08 * dy / n), xytext=(x0, 0),
                arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=1.5))

mu, P = np.linalg.eigh(np.array([[4.0, -1.0], [-1.0, 1.0]]))
s = np.array([-L, L]) * 1.5
ax.plot(s * P[0, 0], s * P[1, 0], ":", color="tab:blue", lw=1.2,
        label="eixo maior das elipses")
ax.plot(s * P[0, 1], s * P[1, 1], ":", color="tab:purple", lw=1.2,
        label="eixo menor das elipses")

ss = np.linspace(-L, L, 2)
ax.plot(ss, ss, "--", color="tab:green", lw=1.2,
        label=r"isóclina $\dot x=0$: $y=x$")
ax.plot(ss, 4 * ss, "--", color="tab:red", lw=1.2,
        label=r"isóclina $\dot y=0$: $y=4x$")

ax.plot(0, 0, "o", color="k", ms=8, zorder=5, label="ponto fixo (centro)")

ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_aspect("equal")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title(r"Ex. 1b: $\dot x=-x+y,\ \dot y=-4x+y$ — centro, $\lambda=\pm i\sqrt{3}$")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
plt.tight_layout()
plt.savefig("retrato_fase_1b.png", dpi=200)
plt.show()
