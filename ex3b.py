import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def f(x, p): # mapa grau 2
    return 1 - p * x**2


def df(x, p): # f'(x)
    return -2 * p * x


def lyapunov(p, x0=0.1, n_trans=2000, n_med=5000, p_max=128, tol=1e-9): # calc lyap, contador de periodos
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


def ciclo(b, x, per, tol=1e-14, itmax=100): # newton em f^p(x) - x, retorna \mu
    for _ in range(itmax):
        y, mu = x, 1.0
        for _ in range(per):
            mu *= df(y, b)
            y = f(y, b)
        dx = (y - x) / (mu - 1)
        x -= dx
        if abs(dx) < tol:
            break
    y, mu = x, 1.0
    for _ in range(per):
        mu *= df(y, b)
        y = f(y, b)
    return x, mu


def mu_de(b, x_guess, per): # multiplicador ciclo
    return ciclo(b, x_guess, per)[1]

N_BIF = 6
b_bif = [0.75]
x_ref = 0.0
for k in range(1, N_BIF): # duplicacoes \Rightarrow mu = -1
    per = 2**k

    passo = (b_bif[-1] - b_bif[-2]) / 4.67 if k > 1 else 0.25
    b = b_bif[-1] + 0.3 * passo
    x = orbita(b, 0.1, 200000)[-1]
    x, mu = ciclo(b, x, per)

    h = passo / 50
    while True:
        b_novo = b + h
        x_novo, mu_novo = ciclo(b_novo, x, per)
        if mu_novo < -1:
            break
        b, x, mu = b_novo, x_novo, mu_novo
    b_n = brentq(lambda bb: mu_de(bb, x, per) + 1, b, b_novo, xtol=1e-15)
    b_bif.append(b_n)

print("Duplicacoes de periodo (mu = -1):")
for k, b_n in enumerate(b_bif):
    per = 2**k
    x_c = orbita(b_n, 0.1, 100000)[-1]
    if per > 1:
        x_c, mu = ciclo(b_n, x_c, per)
    else:
        x_c = (-1 + np.sqrt(1 + 4 * b_n)) / (2 * b_n)
        mu = df(x_c, b_n)
    lam_ciclo = np.log(abs(mu)) / per
    lam_media = lyapunov(b_n, n_trans=20000, n_med=200000)[0]
    print(f"  {per:3d} -> {2*per:3d}: b = {b_n:.10f}, mu = {mu:+.3e}, "
          f"lambda_ciclo = {lam_ciclo:+.2e}, lambda_media = {lam_media:+.2e}")

B0, B1 = 0.5, 1.45
P = np.linspace(B0, B1, 3000) # mapa de bifurcacao
x = np.full_like(P, 0.1)
for _ in range(2000):
    x = f(x, P)
n_pts = 256
pts = np.empty((n_pts, P.size))
for k in range(n_pts):
    x = f(x, P)
    pts[k] = x
LAM = lyapunov(P)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
a1.plot(np.tile(P, n_pts), pts.ravel(), ",", color="k", alpha=0.3)
a1.set_ylabel("$x$")
a1.set_title(r"Rota para o caos por duplicação de período: $x_{n+1}=1-b\,x_n^2$")
a2.plot(P, LAM, color="tab:blue", lw=0.8)
a2.axhline(0, color="k", lw=0.8)
a2.set_ylim(-1.5, 0.4)
a2.set_ylabel(r"$\lambda$")
a2.set_xlabel("$b$")
for k, b_n in enumerate(b_bif[:4]):
    for a in (a1, a2):
        a.axvline(b_n, color="tab:red", lw=0.8, ls="--")
    a1.text(b_n, 1.04, rf"$b_{k+1}$", color="tab:red", ha="center", fontsize=9)
a1.set_ylim(-0.5, 1.1)
plt.tight_layout()
plt.savefig("ex3b_rota_duplicacao.png", dpi=200)

fig, axs = plt.subplots(1, 4, figsize=(16, 4.2)) # lyap qse 0 nas duplicacoes
for k, ax in enumerate(axs):
    b_n = b_bif[k]
    larg = 0.02 * 4.67**(-k) * 4
    bb = np.linspace(b_n - larg, b_n + larg, 401)
    ll = lyapunov(bb, n_trans=20000, n_med=20000)
    ax.plot(bb, ll, color="tab:blue", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(b_n, color="tab:red", lw=0.8, ls="--")
    ax.plot(b_n, 0, "o", color="tab:red", ms=6)
    ax.set_title(rf"${2**k}\to{2**(k+1)}$: $b_{k+1}={b_n:.6f}$", fontsize=10)
    ax.set_xlabel("$b$")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="x", labelrotation=30)
axs[0].set_ylabel(r"$\lambda$")
plt.suptitle(r"3b-i: $\lambda\to 0$ em cada duplicação de período")
plt.tight_layout()
plt.savefig("ex3b_i_zoom_lyapunov.png", dpi=200)

casos = [(0.60, "período 1"), (1.10, "período 2"), (1.32, "período 4"), # series temporais
         (1.385, "período 8"), (1.397, "período 16"), (1.45, "caótico")]
N_TR, N_SH = 50000, 48
fig, axs = plt.subplots(len(casos), 1, figsize=(10, 11), sharex=True)
for ax, (b, rot) in zip(axs, casos):
    xs = orbita(b, 0.1, N_TR + N_SH)[N_TR:]
    lam = lyapunov(b, n_trans=20000, n_med=50000)[0]
    ax.plot(np.arange(N_SH + 1), xs, "o-", ms=3, lw=0.8,
            label=rf"$b={b}$, $\lambda={lam:+.3f}$ ({rot})")
    ax.set_ylabel("$x_n$")
    ax.set_ylim(-1.1, 1.15)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
axs[-1].set_xlabel(rf"$n$ (após descartar {N_TR} iterações de transiente)")
axs[0].set_title("Séries temporais ao longo da cascata de duplicações")
plt.tight_layout()
plt.savefig("ex3b_series.png", dpi=200)
plt.show()
