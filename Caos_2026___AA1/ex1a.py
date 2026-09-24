import os
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagens")
os.makedirs(IMG_DIR, exist_ok=True)

A = np.array([[-3.0, 0.0],
              [3.0, -2.0]]) # inicializando sistema


def campo(x, y):
    return A[0, 0] * x + A[0, 1] * y, A[1, 0] * x + A[1, 1] * y # def. func campo


def solucao(x0, y0, t): # sol geral da EDO 
    x = x0 * np.exp(-3 * t)
    y = (y0 + 3 * x0) * np.exp(-2 * t) - 3 * x0 * np.exp(-3 * t)
    return x, y

lam, V = np.linalg.eig(A) # autovals
print("Autovalores:", lam)
print("Autovetores (colunas, normalizados):\n", V)
print("tr A =", np.trace(A), "| det A =", np.linalg.det(A),
      "| tr^2 - 4 det =", np.trace(A) ** 2 - 4 * np.linalg.det(A)) # check tr, det

L = 2.0
fig, ax = plt.subplots(figsize=(7, 7)) # plot, etc 

xg, yg = np.meshgrid(np.linspace(-L, L, 300), np.linspace(-L, L, 300))
u, v = campo(xg, yg)
ax.streamplot(xg, yg, u, v, color=np.hypot(u, v), cmap="Greys",
              density=1.3, linewidth=0.7, arrowsize=0.8)

s = np.linspace(-L, L, 2)
ax.plot(0 * s, s, color="tab:blue", lw=2.2,
        label=r"$\mathbf{v}_2=(0,1)$, $\lambda_2=-2$ (lenta)")
ax.plot(s, -3 * s, color="tab:red", lw=2.2,
        label=r"$\mathbf{v}_1=(1,-3)$, $\lambda_1=-3$ (rápida)")

t = np.linspace(0, 6, 600)
theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
for th in theta:
    x0, y0 = 1.9 * np.cos(th), 1.9 * np.sin(th)
    xs, ys = solucao(x0, y0, t)
    ax.plot(xs, ys, color="tab:orange", lw=1.2, alpha=0.9)

    k = 40
    ax.annotate("", xy=(xs[k + 1], ys[k + 1]), xytext=(xs[k], ys[k]),
                arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.2))

ax.plot(0, 0, "o", color="k", ms=8, zorder=5, label="ponto fixo")

ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_aspect("equal")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title(r"Ex. 1a: $\dot x=-3x,\ \dot y=3x-2y$")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "retrato_fase_1a.png"), dpi=200)
plt.show()
