import pandas as pd
import cfe.regression as rgsn
import numpy as np
import matplotlib.pyplot as plt
from eep153_tools.sheets import read_sheets


def analyze_good(
    good: str,
    r,
    x0,
    pbar: pd.Series,
    dbar: pd.Series,
    xref: float,
    UseNutrients: list[str],
    nutrient_demand,
    nutrient_adequacy_ratio,
    price_multipliers: np.ndarray = None,
    budget_multipliers: np.ndarray = None,
    figsize: tuple = (12, 16)
):
    """
    - Varies pbar[good] over `price_multipliers` and replots:
        1) Marshallian vs Hicksian demand
        2) Log nutrient adequacy vs Price
        3) Log nutrient adequacy vs Budget
    - Assumes pbar already holds your baseline (and any other shocks).
    """
    # defaults
    if price_multipliers is None:
        price_multipliers = np.geomspace(0.01, 10, 50)
    if budget_multipliers is None:
        budget_multipliers = np.geomspace(0.01, 2, 50)

    # 1) Demand curves
    prices = price_multipliers * pbar[good]
    U0 = r.indirect_utility(x0, pbar)
    q_marshall, q_hicksian = [], []
    for p in prices:
        pvec = pbar.copy(); pvec[good] = p
        q_marshall.append(r.demands(x0, pvec)[good])
        q_hicksian.append(r.demands(U0, pvec, type="Hicksian")[good])

    # 2) Nutrient adequacy vs. Price
    nar_price_dict = {}
    for p in prices:
        pvec = pbar.copy(); pvec[good] = p
        nar = nutrient_adequacy_ratio(xref/4, pvec, dbar)
        nar_price_dict[p] = np.log(nar[UseNutrients])
    nar_vs_price = pd.DataFrame(nar_price_dict).T

    # 3) Nutrient adequacy vs. Budget
    budgets = budget_multipliers * xref
    nar_budget_dict = {}
    for b in budgets:
        nar_b = nutrient_demand(b, pbar)
        nar_budget_dict[b] = np.log(nar_b[UseNutrients])
    nar_vs_budget = pd.DataFrame(nar_budget_dict).T

    # Plot setup
    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)

    # -- Plot 1: Demand Curves --
    ax = axes[0]
    ax.plot(q_marshall, prices, lw=2, label='Marshallian')
    ax.plot(q_hicksian, prices, lw=2, label='Hicksian')
    ax.set_title(f"{good}: Marshallian vs. Hicksian Demand", fontweight='bold')
    ax.set_xlabel(f"Quantity of {good}", fontweight='bold')
    ax.set_ylabel("Price", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, -0.2,
            "Marshallian: demand at fixed income; Hicksian: compensated demand.",
            ha='center', va='top', transform=ax.transAxes, style='italic')

    # -- Plot 2: Nutrient Adequacy vs. Price --
    ax = axes[1]
    for nut in nar_vs_price.columns:
        ax.plot(nar_vs_price[nut], nar_vs_price.index, lw=2, label=nut)
    ax.set_title(f"{good}: Log Nutrient Adequacy vs. Price", fontweight='bold')
    ax.set_xlabel("Log Nutrient Adequacy Ratio", fontweight='bold')
    ax.set_ylabel("Price", fontweight='bold')
    ax.axhline(pbar[good], color='gray', ls='--', lw=1)
    ax.axvline(0, color='gray', ls='--', lw=1)
    ax.legend(title="Nutrient")
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, -0.2,
            "Each curve shows how adequacy changes as the staple's price varies.",
            ha='center', va='top', transform=ax.transAxes, style='italic')

    # -- Plot 3: Nutrient Adequacy vs. Budget --
    ax = axes[2]
    for nut in nar_vs_budget.columns:
        ax.plot(nar_vs_budget[nut], nar_vs_budget.index, lw=2, label=nut)
    ax.set_title(f"{good}: Log Nutrient Adequacy vs. Budget", fontweight='bold')
    ax.set_xlabel("Log Nutrient Adequacy Ratio", fontweight='bold')
    ax.set_ylabel("Budget", fontweight='bold')
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.legend(title="Nutrient")
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, -0.2,
            "Each curve shows how adequacy changes as total budget varies.",
            ha='center', va='top', transform=ax.transAxes, style='italic')

    plt.show()


def nutritional_adequacy_graph(x,pbar,dbar):
    X = np.geomspace(.01*xref,2*xref,100)
    
    pd.DataFrame({x:np.log(nutrient_adequacy_ratio(x,pbar,dbar))[UseNutrients] for x in X}).T.plot()
    plt.legend(UseNutrients)
    plt.xlabel('budget')
    plt.ylabel('log nutrient adequacy ratio')
    plt.axhline(0)
    plt.axvline(xref)
    plt.title('Log Nutrient Adequacy Ratio vs. Budget')

def nutrient_adequacy_ratio(x,p,d,rdi=rdi,days=7):
    hh_rdi = rdi.replace('',0)@d*days

    return nutrient_demand(x,p)/hh_rdi

def nutrient_demand(x,p):
    c = r.demands(x,p)
    fct0,c0 = fct.align(c,axis=0,join='inner')
    N = fct0.T@c0

    N = N.loc[~N.index.duplicated()]
    
    return N


def my_prices(j,p0,p=pbar):
    """
    Change price of jth good to p0, holding other prices fixed at p.
    """
    p = p.copy()
    p.loc[j] = p0
    return p

