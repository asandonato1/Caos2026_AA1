import os
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagens")
os.makedirs(IMG_DIR, exist_ok=True)

NOME_PAR = "b"


def f(x, p): # mapa grau 2
    return 1 - p * x**2


def df(x, p): # f'(x)
    return -2 * p * x


def lyapunov(p, x0=0.1, n_trans=1000, n_med=5000, p_max=128, tol=1e-9): #calc lyap, contador de periodos
    p = np.atleast_1d(np.asarray(p, dtype=float))
    x = np.full_like(p, x0)
    for _ in range(n_trans):
        x = f(x, p)

    orb = [x.copy()]
    for _ in range(p_max):
        orb.append(f(orb[-1], p))
    orb = np.array(orb)
    q = np.zeros(p.size, dtype=int)
    for qq in range(1, p_max + 1):
        ok = (q == 0) & np.all(np.abs(orb[qq:] - orb[:-qq]) < tol, axis=0)
        q[ok] = qq
    n_uso = np.where(q > 0, (n_med // np.maximum(q, 1)) * q, n_med)
    soma = np.zeros_like(p)
    for k in range(n_med):
        soma += np.where(k < n_uso, np.log(np.abs(df(x, p)) + 1e-300), 0.0)
        x = f(x, p)
    return soma / n_uso


def orbita(p, x0, n): # def. orbita
    x = np.empty(n + 1)
    x[0] = x0
    for k in range(n):
        x[k + 1] = f(x[k], p)
    return x

P_PER = 1.10 # periodico 
P_CAO = 1.90 # caos
X0 = 0.10 # def. x0

lam_per = lyapunov(P_PER)[0] # lyap periodico
lam_cao = lyapunov(P_CAO)[0] # lyap caotico
print(f"lambda(b={P_PER}) = {lam_per:.4f}")
print(f"lambda(b={P_CAO}) = {lam_cao:.4f}")

P = np.linspace(1e-3, 2.0, 3000) # mapa de bifurcacao
x = np.full_like(P, X0)
for _ in range(1000):
    x = f(x, P)
n_pts = 200
pts = np.empty((n_pts, P.size))
for k in range(n_pts):
    x = f(x, P)
    pts[k] = x
LAM = lyapunov(P) # lyap(b)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
a1.plot(np.tile(P, n_pts), pts.ravel(), ",", color="k", alpha=0.25)
a1.set_ylabel("$x$")
a1.set_title(r"Mapa de Grau 2: $x_{n+1}=1-b\,x_n^2$")
a2.plot(P, LAM, color="tab:blue", lw=0.7)
a2.axhline(0, color="k", lw=0.8)
a2.set_ylim(-3, 1)
a2.set_ylabel(r"$\lambda$")
a2.set_xlabel("$b$")
for p, cor in ((P_PER, "tab:green"), (P_CAO, "tab:red")):
    for a in (a1, a2):
        a.axvline(p, color=cor, lw=1, ls="--")
a2.text(P_PER, 0.6, f" b={P_PER}", color="tab:green")
a2.text(P_CAO, 0.6, f" b={P_CAO}", color="tab:red", ha="right")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3_bifurcacao_lyapunov.png"), dpi=200)

N = 80
xp = orbita(P_PER, X0, N) # serie periodica
xc = orbita(P_CAO, X0, N) # serie caotica

d2 = np.abs(xp[2:] - xp[:-2])
n_trans = next(n for n in range(d2.size) if np.all(d2[n:] < 1e-4)) # fim do transiente

fig, axs = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
ax = axs[0]
ax.axvspan(0, n_trans, color="0.88", label=f"transiente ($n<{n_trans}$)")
ax.plot(xp, "o-", ms=3, lw=0.8, color="tab:green",
        label=rf"$b={P_PER}$, $\lambda={lam_per:.3f}$ (período 2)")
ax.set_ylabel("$x_n$")
ax.legend(loc="lower right", fontsize=9)
ax = axs[1]
ax.plot(xc, "o-", ms=3, lw=0.8, color="tab:red",
        label=rf"$b={P_CAO}$, $\lambda={lam_cao:.3f}$ (caótico)")
ax.set_ylabel("$x_n$")
ax.set_xlabel("$n$")
ax.set_ylim(-1.35, 1.1)
ax.legend(loc="lower right", fontsize=9)
axs[0].set_title(rf"Séries temporais, $x_0={X0}$")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3a_i_series.png"), dpi=200)

d0 = 1e-10 # sensibilidade as cond. iniciais
N2 = 60
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
for col, (p, lam, cor) in enumerate(((P_PER, lam_per, "tab:green"),
                                     (P_CAO, lam_cao, "tab:red"))):
    x1 = orbita(p, X0, N2)
    x2 = orbita(p, X0 + d0, N2)
    n = np.arange(N2 + 1)
    ax = axs[0, col]
    ax.plot(n, x1, "o-", ms=3, lw=0.8, color=cor, label=rf"$x_0={X0}$")
    ax.plot(n, x2, "s--", ms=3, lw=0.8, color="k", mfc="none",
            label=rf"$x_0={X0}+10^{{-10}}$")
    ax.set_title(rf"$b={p}$, $\lambda={lam:.3f}$")
    ax.set_ylabel("$x_n$")
    ax.legend(fontsize=8, loc="lower left")
    ax = axs[1, col]
    ax.semilogy(n, np.abs(x2 - x1) + 1e-17, "o-", ms=3, lw=0.8, color=cor,
                label=r"$|\delta_n|$")
    ax.semilogy(n, d0 * np.exp(lam * n), "k:", lw=1.5,
                label=r"$|\delta_0|\,e^{\lambda n}$")
    ax.set_ylim(1e-17, 10)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$|\delta_n|=|x_n'-x_n|$")
    ax.legend(fontsize=8, loc="lower right" if lam > 0 else "upper right")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3a_ii_sensibilidade.png"), dpi=200)


def cobweb(ax, p, x0, n_it, lam, cor): # cobweb
    xs = np.linspace(-1, 1, 500)
    ax.plot(xs, f(xs, p), color="k", lw=1.2, label=rf"$f(x)=1-{p}\,x^2$")
    ax.plot(xs, xs, color="0.5", lw=0.8, ls="--", label="$x_{n+1}=x_n$")
    x = x0
    px, py = [x], [0.0]
    for _ in range(n_it):
        y = f(x, p)
        px += [x, y]
        py += [y, y]
        x = y
    ax.plot(px, py, color=cor, lw=1)
    ax.plot(x0, 0, "o", color=cor, ms=5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_n$")
    ax.set_ylabel("$x_{n+1}$")
    ax.set_title(rf"$b={p}$, $\lambda={lam:.3f}$, $x_0={x0}$, {n_it} iterações")
    ax.legend(fontsize=8, loc="lower center")

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

x0_per = round(orbita(P_PER, X0, 200)[-1] + 0.05, 3) # x0 prox. do atrator
cobweb(axs[0], P_PER, x0_per, 10, lam_per, "tab:green")
cobweb(axs[1], P_CAO, X0, 10, lam_cao, "tab:red")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3a_iii_cobweb.png"), dpi=200)
plt.show()
