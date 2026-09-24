import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagens")
os.makedirs(IMG_DIR, exist_ok=True)


def f(x, b): # mapa grau 2
    return 1 - b * x**2


def df(x, b): # f'(x)
    return -2 * b * x


def f_n(x, b, n): # f^n(x)
    for _ in range(n):
        x = f(x, b)
    return x


def lyapunov(b, x0=0.1, n_trans=5000, n_med=20000, p_max=128, tol=1e-9): # calc lyap, contador de periodos
    b = np.atleast_1d(np.asarray(b, dtype=float))
    x = np.full_like(b, x0)
    for _ in range(n_trans):
        x = f(x, b)

    orb = [x.copy()]
    for _ in range(p_max):
        orb.append(f(orb[-1], b))
    orb = np.array(orb)
    q = np.zeros(b.size, dtype=int)
    for qq in range(1, p_max + 1):
        ok = (q == 0) & np.all(np.abs(orb[qq:] - orb[:-qq]) < tol, axis=0)
        q[ok] = qq
    n_uso = np.where(q > 0, (n_med // np.maximum(q, 1)) * q, n_med)
    soma = np.zeros_like(b)
    for k in range(n_med):
        soma += np.where(k < n_uso, np.log(np.abs(df(x, b)) + 1e-300), 0.0)
        x = f(x, b)
    return soma / n_uso


def orbita(b, x0, n): # def. orbita
    x = np.empty(n + 1)
    x[0] = x0
    for k in range(n):
        x[k + 1] = f(x[k], b)
    return x


def ciclo(b, x, per, tol=1e-14, itmax=200): # newton em f^p(x) - x, retorna \mu
    for _ in range(itmax):
        y, mu = x, 1.0
        for _ in range(per):
            mu *= df(y, b)
            y = f(y, b)
        dx = (y - x) / (mu - 1)
        x -= dx
        if abs(dx) < tol:
            break
    y, mu, pts = x, 1.0, []
    for _ in range(per):
        pts.append(y)
        mu *= df(y, b)
        y = f(y, b)
    return np.array(pts), mu


def periodo_atrator(B, n_trans=20000, p_max=64, tol=1e-6): # contador de periodos
    x = np.full_like(B, 0.1)
    for _ in range(n_trans):
        x = f(x, B)
    orb = [x.copy()]
    for _ in range(p_max):
        x = f(x, B)
        orb.append(x.copy())
    orb = np.array(orb)
    per = np.zeros(B.size, dtype=int)
    for p in range(1, p_max + 1):
        ok = (per == 0) & np.all(np.abs(orb[p:] - orb[:-p]) < tol, axis=0)
        per[ok] = p
    return per

B_scan = np.linspace(1e-3, 2.0, 20000) # varredura do periodo
PER = periodo_atrator(B_scan)
m3 = PER == 3
print(f"varredura 0 < b <= 2: atrator de periodo 3 apenas em "
      f"{B_scan[m3].min():.5f} <= b <= {B_scan[m3].max():.5f}  ({m3.sum()} valores de b)")

B_T = 7 / 4
b_in = 1.76
x_in = orbita(b_in, 0.1, 20000)[-1]
pts_in, mu_in = ciclo(b_in, x_in, 3)


def mu3(b): # multiplicador ciclo de periodo 3
    return ciclo(b, x_in, 3)[1]

b_sup = brentq(lambda b: f_n(0.0, b, 3), 1.751, 1.76, xtol=1e-15)

b_dup = brentq(lambda b: mu3(b) + 1, b_sup, 1.775, xtol=1e-15) # fim da janela: \mu_3 = -1

print(f"inicio da janela        = {B_T:.10f}")
print(f"fim do periodo 3 estavel = {b_dup:.10f}")
print(f"b = {b_in}: ciclo = {np.round(np.sort(pts_in), 6)}, mu3 = {mu_in:+.6f}, "
      f"lambda_ciclo = {np.log(abs(mu_in))/3:+.6f}, "
      f"lambda_media = {lyapunov(b_in, n_trans=20000, n_med=200000)[0]:+.6f}")

B0, B1 = 1.70, 1.80
P = np.linspace(B0, B1, 3000) # zoom na janela
x = np.full_like(P, 0.1)
for _ in range(3000):
    x = f(x, P)
n_pts = 300
pts = np.empty((n_pts, P.size))
for k in range(n_pts):
    x = f(x, P)
    pts[k] = x
LAM = lyapunov(P)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
a1.plot(np.tile(P, n_pts), pts.ravel(), ",", color="k", alpha=0.3)
a1.set_ylabel("$x$")
a1.set_title(r"Janela de período 3: $x_{n+1}=1-b\,x_n^2$")
a2.plot(P, LAM, color="tab:blue", lw=0.8)
a2.axhline(0, color="k", lw=0.8)
a2.set_ylim(-1.5, 0.8)
a2.set_ylabel(r"$\lambda$")
a2.set_xlabel("$b$")
for bb, rot, cor in ((B_T, r"$b=7/4$", "tab:red"),
                     (b_dup, r"$b_{3\to6}$", "tab:purple")):
    for a in (a1, a2):
        a.axvline(bb, color=cor, lw=0.9, ls="--")
    a1.text(bb, 1.04, rot, color=cor, ha="center", fontsize=9)
a1.axvspan(B_T, b_dup, color="tab:green", alpha=0.08)
a2.axvspan(B_T, b_dup, color="tab:green", alpha=0.08)
for bb, cor in ((1.745, "tab:orange"), (1.76, "tab:green")):
    a2.plot(bb, lyapunov(bb)[0], "o", color=cor, ms=6)
a1.set_ylim(-1.05, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3c_i_janela.png"), dpi=200)

N_TR, N_SH = 20000, 90
fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True) # series temporais
for ax, (b, cor, rot) in zip(axs, ((1.745, "tab:orange", "caótico"),
                                   (1.76, "tab:green", "período 3"))):
    xs = orbita(b, 0.1, N_TR + N_SH)[N_TR:]
    lam = lyapunov(b, n_trans=20000, n_med=200000)[0]
    ax.plot(np.arange(N_SH + 1), xs, "o-", ms=3, lw=0.8, color=cor,
            label=rf"$b={b}$, $\lambda={lam:+.3f}$ ({rot})")
    ax.set_ylabel("$x_n$")
    ax.set_ylim(-1.45, 1.1)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
axs[1].set_xlabel(rf"$n$ (após descartar {N_TR} iterações de transiente)")
axs[0].set_title("Séries temporais: região caótica e janela de período 3")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3c_i_series.png"), dpi=200)

plt.show()
