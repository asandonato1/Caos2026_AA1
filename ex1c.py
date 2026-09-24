import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def campo(x, y):
    return x * (3 - x - 2 * y), y * (2 - x - y) # def. campo 


def rhs(t, r):
    return campo(r[0], r[1]) # lado direito 


def jac(x, y):
    return np.array([[3 - 2 * x - 2 * y, -2 * x],
                     [-y, 2 - x - 2 * y]]) # jacobiana


def classifica(J):
    tau, delta = np.trace(J), np.linalg.det(J)
    disc = tau**2 - 4 * delta
    if delta < 0:
        return "sela"
    if disc > 0:
        return "estável" if tau < 0 else "instável"
    if disc < 0:
        return "espiral estável" if tau < 0 else ("espiral instável" if tau > 0 else "centro")
    return "degenerado" # classifcações dos ptos

pontos = {"(0,0)": (0.0, 0.0), "(0,2)": (0.0, 2.0),
          "(3,0)": (3.0, 0.0), "(1,1)": (1.0, 1.0)}

autos = {}
for nome, (x0, y0) in pontos.items():
    assert np.allclose(campo(x0, y0), 0), nome
    J = jac(x0, y0)
    lam, V = np.linalg.eig(J)
    autos[nome] = (lam, V)
    print(f"{nome}: J = {J.tolist()}, tr = {np.trace(J):.3f}, "
          f"det = {np.linalg.det(J):.3f}, lambda = {np.round(lam, 5)}, "
          f"-> {classifica(J)}") # classificação 

lam_s, V_s = autos["(1,1)"]
i_est = np.argmin(lam_s.real)
i_ins = np.argmax(lam_s.real)
eps = 1e-4
variedades = []
for i, sentido, cor in [(i_est, -1, "tab:purple"),
                             (i_ins, +1, "tab:red")]:
    for sgn in (+1, -1):
        r0 = np.array([1.0, 1.0]) + sgn * eps * V_s[:, i].real
        sol = solve_ivp(rhs, (0, sentido * (12 if sentido < 0 else 40)), r0, max_step=0.01,
                        rtol=1e-10, atol=1e-12)
        mask = (sol.y[0] > -0.5) & (sol.y[0] < 4) & (sol.y[1] > -0.5) & (sol.y[1] < 3)
        variedades.append((sol.y[0][mask], sol.y[1][mask], cor))

cores_pf = {"sela": "tab:red", "estável": "k", "instável": "white"}

xmin, xmax, ymin, ymax = -0.3, 3.5, -0.3, 2.6
fig, ax = plt.subplots(figsize=(8, 6.2))
xg, yg = np.meshgrid(np.linspace(xmin, xmax, 400), np.linspace(ymin, ymax, 400))
u, v = campo(xg, yg)
ax.streamplot(xg, yg, u, v, color="0.72", density=1.4, linewidth=0.6, arrowsize=0.8)

s = np.linspace(xmin, xmax, 2)
ax.plot(s, (3 - s) / 2, "--", color="tab:green", lw=1.3)
ax.plot(s, 2 - s, "--", color="tab:orange", lw=1.3)
ax.axvline(0, color="tab:green", lw=1.3, ls="--")
ax.axhline(0, color="tab:orange", lw=1.3, ls="--")

for xs, ys, cor in variedades:
    ax.plot(xs, ys, color=cor, lw=2.2)
# plots e anotações 
for nome, (x0, y0) in pontos.items():
    tipo = classifica(jac(x0, y0))
    ax.plot(x0, y0, "o", ms=10, mfc=cores_pf[tipo], mec="k", mew=1.5, zorder=6)
    ax.annotate(f"{nome}\n{tipo}",
                (x0, y0), textcoords="offset points", xytext=(10, 8), fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9))

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title(r"Ex. 1c: $\dot x=x(3-x-2y),\ \dot y=y(2-x-y)$")
plt.tight_layout()
plt.savefig("retrato_fase_1c_global.png", dpi=200)

fig2, axs = plt.subplots(2, 2, figsize=(10, 10))
h = 0.35
for ax, (nome, (x0, y0)) in zip(axs.flat, pontos.items()):
    J = jac(x0, y0)
    lam, V = autos[nome]
    xg, yg = np.meshgrid(np.linspace(x0 - h, x0 + h, 200), np.linspace(y0 - h, y0 + h, 200))
    u, v = campo(xg, yg)
    ax.streamplot(xg, yg, u, v, color="0.55", density=1.3, linewidth=0.7, arrowsize=0.9)
    tipo_pf = classifica(J)
    for k in range(2):
        d = V[:, k].real / np.linalg.norm(V[:, k].real)
        if tipo_pf == "sela":
            cor, papel = ("tab:purple", "estável") if lam[k].real < 0 else ("tab:red", "instável")
        else:
            lenta = abs(lam[k].real) == min(abs(lam.real))
            cor, papel = ("tab:blue", "lenta") if lenta else ("tab:orange", "rápida")
        ax.plot([x0 - 1.2 * h * d[0], x0 + 1.2 * h * d[0]],
                [y0 - 1.2 * h * d[1], y0 + 1.2 * h * d[1]], color=cor, lw=2,
                label=rf"$\lambda={lam[k].real:.4f}$ ({papel}), $\mathbf{{v}}\propto({d[0]:.3f},\,{d[1]:.3f})$")
    tipo = tipo_pf
    ax.plot(x0, y0, "o", ms=9, mfc=cores_pf[classifica(J)], mec="k", mew=1.5, zorder=6)
    ax.set_xlim(x0 - h, x0 + h)
    ax.set_ylim(y0 - h, y0 + h)
    ax.set_aspect("equal")
    ax.set_title(f"{nome}: {tipo}")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
plt.suptitle("Ex. 1c: retratos de fase locais (campo não linear + autovetores de J)")
plt.tight_layout()
plt.savefig("retrato_fase_1c_local.png", dpi=200)
plt.show()
