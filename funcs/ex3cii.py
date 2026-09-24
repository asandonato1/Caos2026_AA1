import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagens")
os.makedirs(IMG_DIR, exist_ok=True)


def f(x, b):
    return 1 - b * x**2


def df(x, b):
    return -2 * b * x


def lyapunov(b, x0=0.1, n_trans=20000, n_med=20000, p_max=128, tol=1e-9):
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


def orbita(b, x0, n):
    x = np.empty(n + 1)
    x[0] = x0
    for k in range(n):
        x[k + 1] = f(x[k], b)
    return x


def ciclo(b, x, per, tol=1e-14, itmax=200):
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


def periodo_atrator(B, n_trans=100000, p_max=96, tol=1e-7):
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

B_scan = np.linspace(1.765, 1.790, 10000)
PER = periodo_atrator(B_scan)
print("Varredura do periodo do atrator (1.765 <= b <= 1.790):")
for p in (3, 6, 12, 24, 48):
    idx = np.flatnonzero(PER == p)
    if idx.size:

        fim = idx[0]
        while fim + 1 < PER.size and PER[fim + 1] == p:
            fim += 1
        print(f"  periodo {p:3d}: {B_scan[idx[0]]:.5f} <= b <= {B_scan[fim]:.5f}")

N_DUP = 5
b_dup = []
b = 1.760
x = orbita(b, 0.1, 100000)[-1]
for k in range(N_DUP):
    per = 3 * 2**k
    x, mu = ciclo(b, x, per)
    h = 1e-4 if k == 0 else (b_dup[-1] - (b_dup[-2] if k > 1 else 1.75)) / 200
    while True:
        x_n, mu_n = ciclo(b + h, x, per)
        if mu_n < -1:
            break
        b, x, mu = b + h, x_n, mu_n
    b_k = brentq(lambda bb: ciclo(bb, x, per)[1] + 1, b, b + h, xtol=1e-15)
    b_dup.append(b_k)

    b = b_k + 0.3 * h
    x = orbita(b, x, 200000)[-1]

print("\nDuplicacoes dentro da janela (mu = -1  =>  lambda = 0):")
for k, b_k in enumerate(b_dup):
    per = 3 * 2**k
    print(f"  {per:3d} -> {2*per:3d}: b = {b_k:.8f}")
b_inf_est = b_dup[-1] + (b_dup[-1] - b_dup[-2]) / (4.669 - 1)
print(f"  acumulacao estimada: b ~ {b_inf_est:.5f}")

B0, B1 = 1.765, 1.790
P = np.linspace(B0, B1, 3000)
x = np.full_like(P, 0.1)
for _ in range(20000):
    x = f(x, P)
n_pts = 384
pts = np.empty((n_pts, P.size))
for k in range(n_pts):
    x = f(x, P)
    pts[k] = x
LAM = lyapunov(P)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
a1.plot(np.tile(P, n_pts), pts.ravel(), ",", color="k", alpha=0.35)
a1.set_ylabel("$x$")
a1.set_title(r"Duplicações de período dentro da janela de período 3: $x_{n+1}=1-b\,x_n^2$")
a2.plot(P, LAM, color="tab:blue", lw=0.8)
a2.axhline(0, color="k", lw=0.8)
a2.set_ylim(-1.0, 0.5)
a2.set_ylabel(r"$\lambda$")
a2.set_xlabel("$b$")
rot = [r"$3\to6$", r"$6\to12$", r"$12\to24$", r"$24\to48$"]
for k in range(4):
    for a in (a1, a2):
        a.axvline(b_dup[k], color="tab:red", lw=0.8, ls="--")
    a1.text(b_dup[k], 1.05, rot[k], color="tab:red", ha="center" if k < 2 else ("right" if k == 2 else "left"), fontsize=8.5)
a1.set_ylim(-0.85, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3c_ii_cascata.png"), dpi=200)


def ramo_central(B, p, n_trans=100000, n_pts=300):
    x = np.full_like(B, 0.1)
    for _ in range(n_trans):
        x = f(x, B)
    fases = [x.copy()]
    for _ in range(p - 1):
        x = f(x, B)
        fases.append(x.copy())
    fases = np.array(fases)
    j0 = np.argmin(np.abs(fases), axis=0)
    x = fases[j0, np.arange(B.size)]
    out = np.empty((n_pts, B.size))
    for k in range(n_pts):
        for _ in range(p):
            x = f(x, B)
        out[k] = x
    return out

rot_dup = [r"$3\to6$", r"$6\to12$", r"$12\to24$", r"$24\to48$", r"$48\to96$"]
b_ant = [1.75] + b_dup[:-1]
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
for k, ax in enumerate(axs.flat):
    esq = b_dup[k] - 0.3 * (b_dup[k] - b_ant[k])
    dir_ = b_inf_est + 0.6 * (b_inf_est - b_dup[k])
    PK = np.linspace(esq, dir_, 2500)
    per = 3 * 2**k
    pts_k = ramo_central(PK, per)
    ax.plot(np.tile(PK, pts_k.shape[0]), pts_k.ravel(), ",", color="k", alpha=0.4)
    lo, hi = np.percentile(pts_k, [0.2, 99.8])
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(esq, dir_)
    for kk in range(k, len(b_dup)):
        ax.axvline(b_dup[kk], color="tab:red", lw=0.8, ls="--")
    ax.axvline(b_inf_est, color="tab:purple", lw=0.9, ls=":")
    y_txt = hi + 0.5 * pad
    for i_t, kk in enumerate(range(k, min(k + 3, len(b_dup)))):
        ax.text(b_dup[kk], y_txt - i_t * 0.9 * pad, rot_dup[kk], color="tab:red", fontsize=8,
                ha="right", va="center")
    ax.text(b_inf_est, lo - 0.5 * pad, r" $b_\infty$", color="tab:purple", fontsize=9,
            va="center")
    ax.set_title(rf"zoom {k+1}: ramo $x\approx0$ amostrado a cada {per} iterações", fontsize=10)
    ax.set_xlabel("$b$")
    ax.set_ylabel("$x$")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.tick_params(axis="x", labelrotation=20)
plt.suptitle("Cascata dentro da janela de período 3: zooms sucessivos com eixos reescalados")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3c_ii_zoom.png"), dpi=200)

bb = np.linspace(b_inf_est + 1e-4, 1.790, 400)
ll = lyapunov(bb)
b_cao = round(float(bb[np.argmax(ll > 0.05)]), 4)
casos = [(1.7660, "período 3"),
         (round((b_dup[0] + b_dup[1]) / 2, 4), "período 6"),
         (round((b_dup[1] + b_dup[2]) / 2, 4), "período 12"),
         (round((b_dup[2] + b_dup[3]) / 2, 5), "período 24"),
         (b_cao, "caótico")]
N_TR, N_SH = 200000, 36
N_R = 48
fig, axs = plt.subplots(len(casos), 2, figsize=(14, 11),
                        gridspec_kw={"width_ratios": [1, 1.25]})
for row, (b, r) in enumerate(casos):
    xs = orbita(b, 0.1, N_TR + 3 * N_R + 3)[N_TR:]
    per = periodo_atrator(np.array([b]))[0]
    lam = lyapunov(b, n_trans=100000, n_med=200000)[0]
    rot_leg = rf"$b={b}$, $\lambda={lam:+.3f}$ ({r})"

    ax = axs[row, 0]
    ax.plot(np.arange(N_SH + 1), xs[:N_SH + 1], "o-", ms=3, lw=0.7, color="tab:blue")
    ax.set_ylim(-1.0, 1.1)
    ax.set_ylabel("$x_n$")
    ax.set_title(rot_leg, fontsize=9.5)

    ax = axs[row, 1]
    k0 = int(np.argmax(np.abs(xs[:3]) < 0.2))
    sub = xs[k0::3][:N_R]
    ax.plot(np.arange(sub.size), sub, "o-", ms=3.5, lw=0.7, color="tab:red")
    ax.set_ylim(-0.1, 0.05)
    ax.set_ylabel(r"$x_{3m}$ (ramo $x\approx0$)")
    ax.set_title(rot_leg, fontsize=9.5)
    print(f"serie b = {b}: periodo detectado = {per if per else 'nenhum <= 96'}, lambda = {lam:+.4f}")
axs[-1, 0].set_xlabel("$n$")
axs[-1, 1].set_xlabel(r"$m$ (retornos ao ramo $x\approx 0$, a cada 3 iterações)")
plt.suptitle(rf"Séries temporais ao longo da cascata (após descartar {N_TR} iterações de transiente)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "ex3c_ii_series.png"), dpi=200)

d0 = 1e-10
N_D = 600
casos_d = [(1.7660, "período 3"), (1.7782, "período 12"),
           (1.77947, "período 24"), (1.7803, "caótico")]
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
for ax, (b, r) in zip(axs.flat, casos_d):
    lam = lyapunov(b, n_trans=100000, n_med=200000)[0]
    xa = orbita(b, 0.1, 200000)[-1]
    x1 = orbita(b, xa, N_D)
    x2 = orbita(b, xa + d0, N_D)
    n = np.arange(N_D + 1)
    dd = np.abs(x2 - x1)
    cor = "tab:red" if lam > 0 else "tab:green"

    n_max = N_D if lam > 0 else int(min(N_D, 1.15 * np.argmax((dd < 1e-16) & (n > 0))))
    ax.semilogy(n[:n_max + 1], dd[:n_max + 1] + 1e-18, "-", lw=0.9, color=cor,
                label=r"$|\delta_n|$ (simulação)")
    ax.semilogy(n[:n_max + 1], d0 * np.exp(lam * n[:n_max + 1]), "k--", lw=1.2,
                label=r"$|\delta_0|\,e^{\lambda n}$")
    ax.axhline(d0, color="0.6", lw=0.6, ls=":")
    ax.set_xlim(0, n_max)
    ax.set_ylim(1e-17, 1)
    ax.set_title(rf"$b={b}$ ({r}),  $\lambda={lam:+.3f}$", fontsize=10.5)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
    ok = (dd > 1e-15) & (dd < 1e-4)
    if lam > 0:
        ok &= n < np.argmax(dd > 1e-4)
    taxa = np.polyfit(n[ok], np.log(dd[ok]), 1)[0] if ok.sum() > 20 else np.nan
    print(f"distanciamento b = {b}: lambda = {lam:+.4f}, taxa ajustada = {taxa:+.4f}")
for ax in axs[1]:
    ax.set_xlabel("$n$")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$|\delta_n| = |x_n' - x_n|$")
from matplotlib.lines import Line2D
h = [Line2D([], [], color="tab:green", lw=1.2), Line2D([], [], color="tab:red", lw=1.2),
     Line2D([], [], color="k", ls="--", lw=1.2)]
l = [r"$|\delta_n|$, $\lambda<0$ (periódico)", r"$|\delta_n|$, $\lambda>0$ (caótico)",
     r"$|\delta_0|\,e^{\lambda n}$"]
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
fig.suptitle(r"Distanciamento de órbitas vizinhas dentro da janela ($|\delta_0| = 10^{-10}$)")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(os.path.join(IMG_DIR, "ex3c_ii_distanciamento.png"), dpi=200)
plt.show()
