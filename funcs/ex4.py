import numpy as np
from scipy.optimize import brentq
import mpmath as mp

mp.mp.dps = 40

MAPAS = { # mapa grau 2
    "Grau 2": dict(f=lambda x, a: 1 - a * x**2,
                   df=lambda x, a: -2 * a * x,
                   fm=lambda x, a: 1 - a * x**2,
                   dfm=lambda x, a: -2 * a * x,
                   a1=0.75, a_ini=0.60, nome_par="b"),
}
N_DUP = 7


def ciclo(m, a, x, p, tol=1e-14, itmax=200): # newton em f^p(x) - x, retorna \mu
    f, df = m["f"], m["df"]
    for _ in range(itmax):
        y, mu = x, 1.0
        for _ in range(p):
            mu *= df(y, a)
            y = f(y, a)
        dx = (y - x) / (mu - 1)
        x -= dx
        if abs(dx) < tol:
            break
    y, mu = x, 1.0
    for _ in range(p):
        mu *= df(y, a)
        y = f(y, a)
    return x, mu


def ciclo_mp(m, a, x, p, itmax=40): # precisao
    f, df = m["fm"], m["dfm"]
    x = mp.mpf(x)
    for _ in range(itmax):
        y, mu = x, mp.mpf(1)
        for _ in range(p):
            mu *= df(y, a)
            y = f(y, a)
        dx = (y - x) / (mu - 1)
        x -= dx
        if abs(dx) < mp.mpf(10) ** (-32):
            break
    y, mu = x, mp.mpf(1)
    for _ in range(p):
        mu *= df(y, a)
        y = f(y, a)
    return x, mu


def orbita(m, a, x, n): # def. orbita
    for _ in range(n):
        x = m["f"](x, a)
    return x


def periodo_minimo_ok(m, a, x, p): # evita ciclo de periodo p/2
    if p == 1:
        return True
    y = orbita(m, a, x, p // 2)
    return abs(y - x) > 1e-8


def duplicacoes(m): # a_n: \mu = -1
    a_n = []
    a = m["a_ini"]
    h = 1e-3
    x = orbita(m, a, 0.1, 100000)
    for k in range(N_DUP):
        p = 2**k
        x, mu = ciclo(m, a, x, p)
        assert periodo_minimo_ok(m, a, x, p), f"Newton convergiu para ciclo de periodo menor (p={p})"
        while True:
            x_novo, mu_novo = ciclo(m, a + h, x, p)
            if mu_novo < -1:
                break
            a, x, mu = a + h, x_novo, mu_novo
        a_k = brentq(lambda aa: ciclo(m, aa, x, p)[1] + 1, a, a + h, xtol=1e-15, rtol=1e-15) # brent

        g = lambda aa: ciclo_mp(m, aa, x, p)[1] + 1
        a_mp = mp.findroot(g, (mp.mpf(a_k), mp.mpf(a_k) + mp.mpf("1e-12")), # refinamento
                           solver="secant", tol=mp.mpf(10) ** (-28))
        a_n.append(a_mp)

        D_est = 0.1 if k == 0 else float(a_n[-1] - a_n[-2]) / 4.67
        a = float(a_mp) + 0.3 * D_est
        h = D_est / 200
        x = orbita(m, a, x, 200000)
    return a_n

for nome, m in MAPAS.items(): # tabela a_n, D_n, \delta_n
    a_n = duplicacoes(m)
    print(f"\n{nome}   ({m['nome_par']}_1 exato = {m['a1']:.10f})")
    print(f"{'bifurcacao':>12} {'a_n':>16} {'D_n':>14} {'delta_n':>10}")
    D = [None]
    for k in range(1, len(a_n)):
        D.append(a_n[k] - a_n[k - 1])
    for k, a in enumerate(a_n):
        p = 2**k
        d_str = f"{float(D[k]):.10f}" if k >= 1 else "---"
        r_str = f"{float(D[k-1] / D[k]):.6f}" if k >= 2 else "---"
        print(f"{p:>5} -> {2*p:<4} {mp.nstr(a, 14):>16} {d_str:>14} {r_str:>10}")
