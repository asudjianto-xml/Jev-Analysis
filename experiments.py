"""Monte Carlo checks of the exact risk and calibration formulas.

Produces validation_sweeps.pdf/.png and results.json. Runs on GPU when available.
"""
import json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

SEED = 20261016
torch.manual_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = torch.float64

# ---------------- parameters (Table 2) ----------------
d, r = 5, 3
beta = torch.tensor([0.9, -0.6, 0.5, 0.2, -0.2], dtype=dt, device=dev)
delta = torch.tensor([0.22, -0.13, 0.10, 0.35, -0.20], dtype=dt, device=dev)
gam = beta + delta
s2, t2, om = 1.3, 0.7, 0.22
NS = [12, 15, 20, 27, 40, 60, 95, 150, 250]
LAM_N = 0.45
N_LAM = 27
LAMS = np.linspace(0, 1, 31)
R = 20000           # training-set replications per configuration
NTEST = 2_000_000   # probit test observations per angle (calibration bins)
NBRIER = 50         # chunks of NTEST draws for the Brier excess (1e8 draws per angle)

# Sigma = I and A = [I_r 0], so the population quantities are coordinate sums.
B = float((delta ** 2).sum())
BA = float((beta[r:] ** 2).sum())
GA = float((gam[r:] ** 2).sum())
SA = float((beta[r:] * gam[r:]).sum())
DA = float((delta[:r] ** 2).sum())
Lnoise = torch.linalg.cholesky(torch.tensor([[s2, om], [om, t2]], dtype=dt, device=dev))


# ---------------- exact formulas ----------------
def full_mix(n, lam):  # Theorem 3.3 / Remark 3.4 (correlated noise)
    C = d / (n - d - 1)
    return lam ** 2 * B + C * ((1 - lam) ** 2 * s2 + lam ** 2 * t2 + 2 * lam * (1 - lam) * om)


def unified(n, lam):  # Theorem 4.2
    C = r / (n - r - 1)
    return BA + lam ** 2 * DA + C * ((1 - lam) ** 2 * (BA + s2) + lam ** 2 * (GA + t2)
                                     + 2 * lam * (1 - lam) * (SA + om))


def lam_full(n):
    C = d / (n - d - 1)
    return min(1, max(0, C * (s2 - om) / (B + C * (s2 + t2 - 2 * om))))


def lam_unified(n):
    C = r / (n - r - 1)
    U, V, H = BA + s2, GA + t2, SA + om
    return min(1, max(0, C * (U - H) / (DA + C * (U + V - 2 * H))))


# ---------------- simulation ----------------
def draw_design(n, R, kind):
    if kind == "gaussian":
        return torch.randn(R, n, d, dtype=dt, device=dev)
    if kind == "uniform":  # unit variance
        return (torch.rand(R, n, d, dtype=dt, device=dev) * 2 - 1) * math.sqrt(3)
    if kind == "t5":  # Student t, 5 df, scaled to unit variance
        z = torch.randn(R, n, d, dtype=dt, device=dev)
        chi = torch.distributions.Chi2(torch.tensor(5.0, dtype=dt, device=dev)).sample((R, n, d))
        return z / torch.sqrt(chi / 5) / math.sqrt(5 / 3)
    raise ValueError(kind)


def ls(Xm, y):
    G = Xm.transpose(1, 2) @ Xm
    return torch.linalg.solve(G, Xm.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1)


def simulate(n, lams, R, kind="gaussian"):
    """Per-replication excess risk for each lambda; full and compressed heads.

    With Sigma = I the test expectation is analytic given the fitted coefficients,
    so the Monte Carlo is over training sets only."""
    X = draw_design(n, R, kind)
    E = torch.randn(R, n, 2, dtype=dt, device=dev) @ Lnoise.T
    Y = X @ beta + E[..., 0]
    T = X @ gam + E[..., 1]
    out = {"full": [], "comp": []}
    for lam in lams:
        V = (1 - lam) * Y + lam * T
        bf = ls(X, V)
        wc = ls(X[..., :r], V)
        out["full"].append(((bf - beta) ** 2).sum(1))
        out["comp"].append(((wc - beta[:r]) ** 2).sum(1) + BA)
    return {k: torch.stack(v).cpu().numpy() for k, v in out.items()}


def mean_ci(x):
    m = x.mean(-1)
    h = 1.96 * x.std(-1, ddof=1) / math.sqrt(x.shape[-1])
    return m, h


res = {"params": dict(d=d, r=r, B=B, B_A=BA, D_A=DA, G_A=GA, S_A=SA, sigma2=s2, tau2=t2,
                      omega=om, R=R, seed=SEED)}

# (a) sample-size sweep
sweep = {"n": NS, "ols": [], "teacher": [], "comp": [], "unified": []}
for n in NS:
    s = simulate(n, [0.0, 1.0, LAM_N], R)
    for key, arr in [("ols", s["full"][0]), ("teacher", s["full"][1]),
                     ("comp", s["comp"][0]), ("unified", s["comp"][2])]:
        m, h = mean_ci(arr)
        sweep[key].append((float(m), float(h)))
th_a = {"ols": lambda n: full_mix(n, 0), "teacher": lambda n: full_mix(n, 1),
        "comp": lambda n: unified(n, 0), "unified": lambda n: unified(n, LAM_N)}
res["sweep"] = sweep
res["sweep_theory"] = {k: [f(n) for n in NS] for k, f in th_a.items()}
# crossover sample sizes from the closed-form criteria
res["crossover_teacher_n"] = d + 1 + d * (s2 - t2) / B
res["crossover_comp_n"] = d + 1 + s2 * (d - r) / BA

# (b) teacher-weight sweep at n = 27
s = simulate(N_LAM, LAMS, R)
mb_full, hb_full = mean_ci(s["full"])
mb_comp, hb_comp = mean_ci(s["comp"])
res["lam_sweep"] = dict(lams=LAMS.tolist(), full=mb_full.tolist(), full_ci=hb_full.tolist(),
                        comp=mb_comp.tolist(), comp_ci=hb_comp.tolist(),
                        lam_full=lam_full(N_LAM), lam_comp=lam_unified(N_LAM),
                        lam_full_mc=float(LAMS[mb_full.argmin()]),
                        lam_comp_mc=float(LAMS[mb_comp.argmin()]))

# (d) departures from Gaussian design: equicorrelated Sigma, three laws with the same
# covariance. A selects the first r coordinates, so X'u_beta is uncorrelated with Z
# under every law but independent of Z only under the Gaussian law.
RHO = 0.5
Sig = (1 - RHO) * torch.eye(d, dtype=dt, device=dev) + RHO * torch.ones(d, d, dtype=dt, device=dev)
Lsig = torch.linalg.cholesky(Sig)
CA = Sig[:r, :r]
wstar = torch.linalg.solve(CA, Sig[:r] @ beta)
wgam = torch.linalg.solve(CA, Sig[:r] @ gam)
ub = beta.clone(); ub[:r] -= wstar
ug = gam.clone(); ug[:r] -= wgam
BA2 = float(ub @ Sig @ ub); GA2 = float(ug @ Sig @ ug); SA2 = float(ub @ Sig @ ug)
DA2 = float((wgam - wstar) @ CA @ (wgam - wstar))


def unified2(n, lam):
    C = r / (n - r - 1)
    return BA2 + lam ** 2 * DA2 + C * ((1 - lam) ** 2 * (BA2 + s2) + lam ** 2 * (GA2 + t2)
                                       + 2 * lam * (1 - lam) * (SA2 + om))


def draw_corr(n, R, kind):
    W = torch.randn(R, n, d, dtype=dt, device=dev)
    if kind == "uniform":
        W = (torch.rand(R, n, d, dtype=dt, device=dev) * 2 - 1) * math.sqrt(3)
    elif kind == "mvt5":  # multivariate t, 5 df, unit-variance scale mixture
        chi = torch.distributions.Chi2(torch.tensor(5.0, dtype=dt, device=dev)).sample((R, n, 1))
        W = W / torch.sqrt(chi / 5) / math.sqrt(5 / 3)
    return W @ Lsig.T


robust = {"rho": RHO, "B_A": BA2, "D_A": DA2, "G_A": GA2, "S_A": SA2}
for kind in ["gaussian", "uniform", "mvt5"]:
    robust[kind] = {}
    for n in [12, 27, 95]:
        X = draw_corr(n, R, kind)
        E = torch.randn(R, n, 2, dtype=dt, device=dev) @ Lnoise.T
        Y = X @ beta + E[..., 0]; T = X @ gam + E[..., 1]
        bf = ls(X, Y)
        ef = torch.einsum("bi,ij,bj->b", bf - beta, Sig, bf - beta)
        wc = ls(X[..., :r], (1 - LAM_N) * Y + LAM_N * T)
        wl = (1 - LAM_N) * wstar + LAM_N * wgam
        ec = torch.einsum("bi,ij,bj->b", wc - wl, CA, wc - wl) + BA2 + LAM_N ** 2 * DA2
        mo, ho = mean_ci(ef.cpu().numpy()); mu, hu = mean_ci(ec.cpu().numpy())
        robust[kind][n] = dict(ols=float(mo / full_mix(n, 0)), ols_ci=float(ho / full_mix(n, 0)),
                               unified=float(mu / unified2(n, LAM_N)),
                               unified_ci=float(hu / unified2(n, LAM_N)))
res["robust"] = robust


# (c) probit calibration under rotation
def Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def brier_excess(va, vb, c):  # Theorem 5.1
    return (math.asin(va / (1 + va)) + math.asin(vb / (1 + vb))
            - 2 * math.asin(c / math.sqrt((1 + va) * (1 + vb)))) / (2 * math.pi)


def brier_residual(va, vb, c):  # Corollary 5.6
    return (math.asin(va / (1 + va)) - math.asin(c * c / (vb * (1 + va)))) / (2 * math.pi)


a = torch.tensor([1.0, 0.0], dtype=dt, device=dev)
cal = {}
edges = torch.linspace(0, 1, 21, dtype=dt, device=dev)
for deg in [0, 30, 60]:
    th = math.radians(deg)
    b = torch.tensor([math.cos(th), math.sin(th)], dtype=dt, device=dev)
    X = torch.randn(NTEST, 2, dtype=dt, device=dev)
    eta = Phi(X @ a)
    Yb = (torch.rand(NTEST, dtype=dt, device=dev) < eta).to(dt)
    s_ = X @ b
    P = Phi(s_)
    va, vb, c = 1.0, 1.0, math.cos(th)
    kap = (c / vb) / math.sqrt(1 + va - c * c / vb)
    Pcal = Phi(kap * s_)
    idx = torch.bucketize(P, edges[1:-1])
    pm, ym, yh = [], [], []
    for k in range(20):
        m = idx == k
        cnt = int(m.sum())
        if cnt < 2000:
            continue
        pm.append(float(P[m].mean()))
        yk = float(Yb[m].mean())
        ym.append(yk)
        yh.append(1.96 * math.sqrt(yk * (1 - yk) / cnt))
    cal[deg] = dict(kappa=kap, p=pm, y=ym, ci=yh,
                    raw_th=brier_excess(va, vb, c), rec_th=brier_residual(va, vb, c))
    # Brier excesses from NBRIER chunks of NTEST draws; chunk means give the standard error
    raw, rec = [], []
    for _ in range(NBRIER):
        Xc = torch.randn(NTEST, 2, dtype=dt, device=dev)
        ec, sc = Phi(Xc @ a), Xc @ b
        raw.append(float(((Phi(sc) - ec) ** 2).mean()))
        rec.append(float(((Phi(kap * sc) - ec) ** 2).mean()))
    cal[deg].update(raw_mc=float(np.mean(raw)), raw_se=float(np.std(raw, ddof=1) / math.sqrt(NBRIER)),
                    rec_mc=float(np.mean(rec)), rec_se=float(np.std(rec, ddof=1) / math.sqrt(NBRIER)))
res["calibration"] = cal

with open("results.json", "w") as f:
    json.dump(res, f, indent=1)

# ---------------- figure ----------------
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED = "#222222", "#6b6b6b"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.6})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))

ng = np.geomspace(min(NS), max(NS), 200)
labels = {"ols": "Outcome OLS, full", "teacher": "Teacher OLS, full",
          "comp": "Outcome OLS, compressed", "unified": rf"Mixed $\lambda={LAM_N}$, compressed"}
marks = {"ols": "o", "teacher": "s", "comp": "^", "unified": "D"}
for i, k in enumerate(["ols", "teacher", "comp", "unified"]):
    ax[0].plot(ng, [th_a[k](n) for n in ng], color=COL[i], label=labels[k])
    m = np.array([v[0] for v in sweep[k]]); h = np.array([v[1] for v in sweep[k]])
    ax[0].errorbar(NS, m, yerr=h, fmt=marks[k], ms=4, color=COL[i], mec="white", mew=0.6,
                   elinewidth=1, capsize=0)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xticks([12, 20, 40, 100, 250]); ax[0].set_xticklabels(["12", "20", "40", "100", "250"])
ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_xlabel("training size $n$"); ax[0].set_ylabel("excess prediction risk")
ax[0].set_title("(a) Risk versus sample size", fontsize=9, color=INK, loc="left")
ax[0].legend(frameon=False, fontsize=7)

lg = np.linspace(0, 1, 200)
ls_ = res["lam_sweep"]
for i, (k, f, lstar, lab, mk) in enumerate([
        ("full", full_mix, ls_["lam_full"], "Full representation", "o"),
        ("comp", unified, ls_["lam_comp"], "Compressed, $r=3$", "^")]):
    ax[1].plot(lg, [f(N_LAM, l) for l in lg], color=COL[i], label=lab)
    ax[1].errorbar(LAMS, ls_[k], yerr=ls_[k + "_ci"], fmt=mk, ms=3.5, color=COL[i],
                   mec="white", mew=0.5, elinewidth=1, capsize=0)
    ax[1].axvline(lstar, color=COL[i], ls=":", lw=1)
ax[1].set_xlabel(r"teacher weight $\lambda$"); ax[1].set_ylabel("excess prediction risk")
ax[1].set_title(f"(b) Mixed supervision, $n={N_LAM}$", fontsize=9, color=INK, loc="left")
ax[1].legend(frameon=False, fontsize=7)

pg = np.linspace(0.005, 0.995, 300)
from scipy.stats import norm
ax[2].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
for i, deg in enumerate([0, 30, 60]):
    kap = cal[deg]["kappa"]
    ax[2].plot(pg, norm.cdf(kap * norm.ppf(pg)), color=COL[i], label=rf"$\theta={deg}^\circ$")
    ax[2].errorbar(cal[deg]["p"], cal[deg]["y"], yerr=cal[deg]["ci"], fmt="o", ms=3.5,
                   color=COL[i], mec="white", mew=0.5, elinewidth=1, capsize=0)
ax[2].set_xlabel("teacher forecast $p$"); ax[2].set_ylabel(r"$\Pr(Y=1\mid P=p)$")
ax[2].set_title("(c) Calibration under rotation", fontsize=9, color=INK, loc="left")
ax[2].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig("validation_sweeps.pdf"); fig.savefig("validation_sweeps.png", dpi=200)

print(json.dumps({k: res[k] for k in ["params", "crossover_teacher_n", "crossover_comp_n"]}, indent=1))
print("lam", {k: ls_[k] for k in ["lam_full", "lam_full_mc", "lam_comp", "lam_comp_mc"]})
for k in ["ols", "teacher", "comp", "unified"]:
    print(k, [(n, round(m, 4), round(t, 4)) for n, (m, h), t in zip(NS, sweep[k], res["sweep_theory"][k])])
print("robust", json.dumps(robust, indent=1))
for deg in cal:
    c_ = cal[deg]
    print(deg, "kappa", round(c_["kappa"], 4), "raw", c_["raw_mc"], c_["raw_se"], c_["raw_th"], "rec", c_["rec_mc"], c_["rec_se"], c_["rec_th"])
