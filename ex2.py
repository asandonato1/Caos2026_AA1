import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import ellipk

s6 = np.sqrt(6.0)


def V1(th): # V_g (\theta)
    return 1 - np.cos(th)


def V2(th): # V_f(theta)
    return th**2 / 2 - th**4 / 24

for th0 in (0.0, s6, -s6):
    J = np.array([[0.0, 1.0], [-1 + th0**2 / 2, 0.0]])
    print(f"theta* = {th0:+.5f}: autovalores = {np.round(np.linalg.eigvals(J), 6)}") # autovals


def T_pendulo(A):
    return 4 * ellipk(np.sin(A / 2)**2) # periodo da eq. nao-aprox


def T_aprox(A):
    f = lambda phi: 1 / np.sqrt(1 - A**2 * (1 + np.sin(phi)**2) / 12)
    return 4 * quad(f, 0, np.pi / 2)[0] # T_aprox
#plots
fig, axs = plt.subplots(1, 3, figsize=(17, 5.4),
                        gridspec_kw={"width_ratios": [1.35, 1, 0.9]})

ax = axs[0]
th = np.linspace(-2 * np.pi - 0.6, 2 * np.pi + 0.6, 600)
om = np.linspace(-3.2, 3.2, 500)
TH, OM = np.meshgrid(th, om)
E1 = OM**2 / 2 + V1(TH)
ax.contour(TH, OM, E1, levels=[0.1, 0.4, 0.9, 1.5, 2.6, 3.5, 4.5],
           colors="tab:blue", linewidths=1)
ax.streamplot(TH, OM, OM, -np.sin(TH), color="0.8", density=0.9,
              linewidth=0.5, arrowsize=0.7)
for n in range(-2, 3):
    ax.plot(n * np.pi, 0, "o", ms=8, mec="k",
            mfc="k" if n % 2 == 0 else "tab:red", zorder=5)
ax.set_xticks(np.arange(-2, 3) * np.pi)
ax.set_xticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
ax.set_xlim(th[0], th[-1])
ax.set_title(r"(1) $\ddot\theta=-\sin\theta$")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\omega=\dot\theta$")

ax = axs[1]
th = np.linspace(-4.2, 4.2, 600)
om = np.linspace(-3.2, 3.2, 500)
TH, OM = np.meshgrid(th, om)
E2 = OM**2 / 2 + V2(TH)
dentro = np.abs(TH) < s6
ax.contour(TH, OM, np.where(dentro, E2, np.nan), levels=[0.1, 0.4, 0.8, 1.2],
           colors="tab:blue", linewidths=1)
ax.contour(TH, OM, E2, levels=[1.8, 2.6, 3.6], colors="tab:green", linewidths=1)
ax.contour(TH, OM, np.where(dentro, np.nan, E2), levels=[-1.5, 0.0, 0.8],
           colors="tab:purple", linewidths=1, linestyles="solid")
ax.streamplot(TH, OM, OM, -TH + TH**3 / 6, color="0.8", density=0.9,
              linewidth=0.5, arrowsize=0.7)
ax.plot(0, 0, "o", ms=8, mfc="k", mec="k", zorder=5)
for th0 in (s6, -s6):
    ax.plot(th0, 0, "o", ms=8, mfc="tab:red", mec="k", zorder=5)
ax.set_xlim(th[0], th[-1])
ax.set_ylim(om[0], om[-1])
ax.set_title(r"(2) $\ddot\theta=-\theta+\theta^3/6$")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\omega=\dot\theta$")

ax = axs[2]
A1 = np.linspace(0.01, np.pi - 1e-3, 400)
A2 = np.linspace(0.01, s6 - 1e-4, 400)
ax.plot(A1, [T_pendulo(a) for a in A1], color="tab:blue", label=r"(1) exato, $A\to\pi$")
ax.plot(A2, [T_aprox(a) for a in A2], color="tab:orange", label=r"(2) aprox., $A\to\sqrt{6}$")
ax.axhline(2 * np.pi, color="k", lw=0.8, ls=":", label=r"$2\pi$ (linear)")
ax.axvline(np.pi, color="tab:blue", lw=0.8, ls="--")
ax.axvline(s6, color="tab:orange", lw=0.8, ls="--")
ax.set_ylim(6, 20)
ax.set_xlabel("amplitude $A$")
ax.set_ylabel("período $T$")
ax.set_title("Período das oscilações")
ax.legend(fontsize=8.5, loc="upper left")

plt.tight_layout()
plt.savefig("diagrama_fases_ex2.png", dpi=200)

print("\n  A      T_exato    T_aprox")
for A in (0.2, 0.5, 1.0, 1.5, 2.0, 2.4):
    print(f"{A:4.1f}   {T_pendulo(A):8.5f}   {T_aprox(A):8.5f}")
plt.show()
