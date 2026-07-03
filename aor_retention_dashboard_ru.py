import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Tuple

COIN_COST_EUR = 1.0  # 1 Coin = 1 freebet = EUR1 bonus budget


@dataclass
class ModelInputs:
    total_bets: int
    turnover: float
    ggr: float
    active_users: int
    n_simulations: int
    n_iterations: int
    target_share: float
    activation_rate: float
    completion_rate: float
    avg_missions_per_completed_user_per_iteration: float
    max_missions_per_completed_user_per_iteration: int
    bet_count_uplift: float
    avg_stake_uplift: float
    hold_uplift: float
    reward_budget_share: float
    freebet_spend_rate_per_iteration: float
    user_bet_cv: float
    iteration_volatility: float
    random_seed: int


def safe_divide(a, b, default=0.0):
    return default if b == 0 else a / b


def pct(x):
    return "n/a" if pd.isna(x) or np.isinf(x) else f"{x * 100:,.1f}%"


def eur(x):
    return "n/a" if pd.isna(x) or np.isinf(x) else f"€{x:,.0f}"


def number(x, decimals=2):
    return "n/a" if pd.isna(x) or np.isinf(x) else f"{x:,.{decimals}f}"


def gaussian_user_weights(rng, active_users, cv):
    raw = rng.normal(loc=1.0, scale=cv, size=active_users)
    weights = np.clip(raw, 0.01, None)
    return weights / weights.mean()


def iteration_weights(rng, n_iterations, volatility):
    raw = rng.normal(loc=1.0, scale=volatility, size=n_iterations)
    weights = np.clip(raw, 0.05, None)
    return weights / weights.sum()


def sample_completed_segment(rng, weights, active_users, target_share, activation_rate, completion_rate):
    targeted = int(rng.binomial(active_users, target_share))
    activated = int(rng.binomial(targeted, activation_rate))
    completed = int(rng.binomial(activated, completion_rate))
    if completed <= 0:
        return targeted, activated, completed, 0.0
    completed_weight_sum = float(rng.choice(weights, size=completed, replace=False).sum())
    return targeted, activated, completed, completed_weight_sum


def simulate_missions(rng, completed_users, avg_missions, max_missions):
    if completed_users <= 0:
        return 0, 0.0
    lam = max(avg_missions - 1.0, 0.0)
    counts = 1 + rng.poisson(lam=lam, size=completed_users)
    counts = np.clip(counts, 1, max_missions)
    total = int(counts.sum())
    return total, safe_divide(total, completed_users)


def run_single_simulation(rng, inp: ModelInputs, sim_id: int) -> Tuple[Dict, pd.DataFrame]:
    avg_stake = safe_divide(inp.turnover, inp.total_bets)
    hold = safe_divide(inp.ggr, inp.turnover)
    weights = gaussian_user_weights(rng, inp.active_users, inp.user_bet_cv)
    iter_weights = iteration_weights(rng, inp.n_iterations, inp.iteration_volatility)

    freebet_balance = 0.0  # на старте у пользователей 0 фрибетов
    rows = []

    for i, w in enumerate(iter_weights, start=1):
        base_bets = inp.total_bets * w
        base_turnover = inp.turnover * w
        base_ggr = inp.ggr * w

        targeted, activated, completed, completed_weight_sum = sample_completed_segment(
            rng, weights, inp.active_users, inp.target_share, inp.activation_rate, inp.completion_rate
        )

        completed_base_bets = base_bets * safe_divide(completed_weight_sum, inp.active_users)
        non_completed_base_bets = base_bets - completed_base_bets
        completed_base_turnover = completed_base_bets * avg_stake
        non_completed_base_turnover = non_completed_base_bets * avg_stake

        bet_mult = 1.0 + inp.bet_count_uplift
        stake_mult = 1.0 + inp.avg_stake_uplift
        hold_mult = 1.0 + inp.hold_uplift

        aor_bets = non_completed_base_bets + completed_base_bets * bet_mult
        aor_turnover = non_completed_base_turnover + completed_base_turnover * bet_mult * stake_mult
        aor_ggr_gross = (
            non_completed_base_turnover * hold
            + completed_base_turnover * bet_mult * stake_mult * hold * hold_mult
        )
        inc_ggr = aor_ggr_gross - base_ggr

        missions, avg_missions_actual = simulate_missions(
            rng,
            completed,
            inp.avg_missions_per_completed_user_per_iteration,
            inp.max_missions_per_completed_user_per_iteration,
        )

        max_reward_budget = max(0.0, inc_ggr)
        planned_reward_budget = max_reward_budget * inp.reward_budget_share
        coins_issued = np.floor(planned_reward_budget / COIN_COST_EUR) if missions > 0 else 0.0
        freebets_issued = coins_issued

        balance_start = freebet_balance
        balance_after_issue = balance_start + freebets_issued
        freebets_spent = balance_after_issue * inp.freebet_spend_rate_per_iteration
        balance_end = balance_after_issue - freebets_spent
        freebet_balance = balance_end

        cost_spent = freebets_spent * COIN_COST_EUR
        cost_issued = freebets_issued * COIN_COST_EUR
        net_after_spent = inc_ggr - cost_spent
        conservative_net = inc_ggr - cost_issued

        rows.append({
            "simulation": sim_id,
            "iteration": i,
            "iteration_weight": w,
            "targeted_users": targeted,
            "activated_users": activated,
            "completed_users": completed,
            "completed_missions": missions,
            "avg_missions_per_completed_user": avg_missions_actual,
            "baseline_bets": base_bets,
            "aor_bets": aor_bets,
            "incremental_bets": aor_bets - base_bets,
            "baseline_turnover": base_turnover,
            "aor_turnover": aor_turnover,
            "incremental_turnover": aor_turnover - base_turnover,
            "baseline_ggr": base_ggr,
            "aor_ggr_gross": aor_ggr_gross,
            "incremental_ggr_gross": inc_ggr,
            "max_reward_budget": max_reward_budget,
            "planned_reward_budget": planned_reward_budget,
            "coins_issued": coins_issued,
            "coins_per_completed_mission": safe_divide(coins_issued, missions),
            "freebets_issued": freebets_issued,
            "freebets_spent": freebets_spent,
            "freebet_balance_start": balance_start,
            "freebet_balance_after_issue": balance_after_issue,
            "freebet_balance_end": balance_end,
            "program_cost_spent": cost_spent,
            "program_cost_issued": cost_issued,
            "net_ggr_after_spent": net_after_spent,
            "conservative_net_ggr": conservative_net,
            "reward_limit_ok": cost_issued <= max(0.0, inc_ggr),
            "break_even_after_spent": net_after_spent >= 0,
            "break_even_conservative": conservative_net >= 0,
        })

    iter_df = pd.DataFrame(rows)
    baseline_ggr = float(iter_df["baseline_ggr"].sum())
    inc_ggr_total = float(iter_df["incremental_ggr_gross"].sum())
    cost_spent_total = float(iter_df["program_cost_spent"].sum())
    cost_issued_total = float(iter_df["program_cost_issued"].sum())
    missions_total = float(iter_df["completed_missions"].sum())
    completed_total = float(iter_df["completed_users"].sum())
    freebets_issued_total = float(iter_df["freebets_issued"].sum())
    freebets_spent_total = float(iter_df["freebets_spent"].sum())
    final_balance = float(iter_df["freebet_balance_end"].iloc[-1])

    sim = {
        "simulation": sim_id,
        "completed_users_total": completed_total,
        "completed_missions_total": missions_total,
        "avg_missions_per_completed_user": safe_divide(missions_total, completed_total),
        "baseline_bets": float(iter_df["baseline_bets"].sum()),
        "aor_bets": float(iter_df["aor_bets"].sum()),
        "incremental_bets": float(iter_df["incremental_bets"].sum()),
        "baseline_turnover": float(iter_df["baseline_turnover"].sum()),
        "aor_turnover": float(iter_df["aor_turnover"].sum()),
        "incremental_turnover": float(iter_df["incremental_turnover"].sum()),
        "baseline_ggr": baseline_ggr,
        "aor_ggr_gross": float(iter_df["aor_ggr_gross"].sum()),
        "incremental_ggr_gross": inc_ggr_total,
        "gross_uplift_pct": safe_divide(inc_ggr_total, baseline_ggr, default=np.nan),
        "max_reward_budget_total": float(iter_df["max_reward_budget"].sum()),
        "planned_reward_budget_total": float(iter_df["planned_reward_budget"].sum()),
        "coins_issued_total": freebets_issued_total,
        "coins_issued_per_active_user": safe_divide(freebets_issued_total, inp.active_users),
        "coins_per_completed_mission": safe_divide(freebets_issued_total, missions_total),
        "freebets_issued_total": freebets_issued_total,
        "freebets_spent_total": freebets_spent_total,
        "final_freebet_balance": final_balance,
        "freebets_issued_per_active_user": safe_divide(freebets_issued_total, inp.active_users),
        "freebets_spent_per_active_user": safe_divide(freebets_spent_total, inp.active_users),
        "final_freebet_balance_per_active_user": safe_divide(final_balance, inp.active_users),
        "program_cost_spent": cost_spent_total,
        "program_cost_issued": cost_issued_total,
        "net_ggr_after_spent": inc_ggr_total - cost_spent_total,
        "conservative_net_ggr": inc_ggr_total - cost_issued_total,
        "roi_after_spent": safe_divide(inc_ggr_total - cost_spent_total, cost_spent_total, default=np.nan),
        "roi_conservative": safe_divide(inc_ggr_total - cost_issued_total, cost_issued_total, default=np.nan),
        "net_uplift_pct_after_spent": safe_divide(inc_ggr_total - cost_spent_total, baseline_ggr, default=np.nan),
        "net_uplift_pct_conservative": safe_divide(inc_ggr_total - cost_issued_total, baseline_ggr, default=np.nan),
        "reward_limit_compliance_share": float(iter_df["reward_limit_ok"].mean()),
        "break_even_after_spent_share": float(iter_df["break_even_after_spent"].mean()),
        "break_even_conservative_share": float(iter_df["break_even_conservative"].mean()),
    }
    return sim, iter_df


@st.cache_data(show_spinner=False)
def run_simulations(inp_dict: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inp = ModelInputs(**inp_dict)
    rng = np.random.default_rng(inp.random_seed)
    sim_rows, iter_frames = [], []
    for sim_id in range(1, inp.n_simulations + 1):
        sim, iters = run_single_simulation(rng, inp, sim_id)
        sim_rows.append(sim)
        iter_frames.append(iters)
    return pd.DataFrame(sim_rows), pd.concat(iter_frames, ignore_index=True)


def build_summary(df):
    order = [
        "completed_users_total", "completed_missions_total", "avg_missions_per_completed_user",
        "incremental_bets", "incremental_turnover", "incremental_ggr_gross", "gross_uplift_pct",
        "max_reward_budget_total", "planned_reward_budget_total",
        "coins_issued_total", "coins_issued_per_active_user", "coins_per_completed_mission",
        "freebets_issued_total", "freebets_spent_total", "final_freebet_balance",
        "freebets_issued_per_active_user", "freebets_spent_per_active_user", "final_freebet_balance_per_active_user",
        "program_cost_spent", "program_cost_issued", "net_ggr_after_spent", "conservative_net_ggr",
        "roi_after_spent", "roi_conservative", "net_uplift_pct_after_spent", "net_uplift_pct_conservative",
    ]
    rows = []
    for m in order:
        s = df[m].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"metric": m, "p05": s.quantile(0.05), "mean": s.mean(), "median": s.median(), "p95": s.quantile(0.95)})
    return pd.DataFrame(rows)


def format_summary(summary):
    names = {
        "completed_users_total": "Completed users суммарно по итерациям",
        "completed_missions_total": "Выполненные миссии всего",
        "avg_missions_per_completed_user": "Миссий на completed user",
        "incremental_bets": "Дополнительные ставки",
        "incremental_turnover": "Дополнительный оборот",
        "incremental_ggr_gross": "Изменение GGR / дополнительный gross GGR",
        "gross_uplift_pct": "Изменение GGR, % к baseline",
        "max_reward_budget_total": "Максимальный фонд Coins / фрибетов",
        "planned_reward_budget_total": "Плановый фонд Coins / фрибетов",
        "coins_issued_total": "Выдано Coins всего",
        "coins_issued_per_active_user": "Coins на active user",
        "coins_per_completed_mission": "Coins на выполненную миссию",
        "freebets_issued_total": "Фрибеты выданы всего",
        "freebets_spent_total": "Фрибеты использованы всего",
        "final_freebet_balance": "Фрибеты на конец симуляции",
        "freebets_issued_per_active_user": "Фрибеты выданы на active user",
        "freebets_spent_per_active_user": "Фрибеты использованы на active user",
        "final_freebet_balance_per_active_user": "Остаток фрибетов на active user",
        "program_cost_spent": "Стоимость использованных фрибетов",
        "program_cost_issued": "Консервативная стоимость выданных фрибетов",
        "net_ggr_after_spent": "Net GGR после использованных фрибетов",
        "conservative_net_ggr": "Консервативный Net GGR после выданных фрибетов",
        "roi_after_spent": "ROI по использованным фрибетам",
        "roi_conservative": "Консервативный ROI по выданным фрибетам",
        "net_uplift_pct_after_spent": "Net uplift после использованных фрибетов",
        "net_uplift_pct_conservative": "Консервативный net uplift после выданных фрибетов",
    }
    money = {"incremental_turnover", "incremental_ggr_gross", "max_reward_budget_total", "planned_reward_budget_total", "program_cost_spent", "program_cost_issued", "net_ggr_after_spent", "conservative_net_ggr"}
    perc = {"gross_uplift_pct", "roi_after_spent", "roi_conservative", "net_uplift_pct_after_spent", "net_uplift_pct_conservative"}
    out = summary.copy()
    out["source_metric"] = out["metric"]
    out["metric"] = out["metric"].map(names).fillna(out["metric"])
    for col in ["p05", "mean", "median", "p95"]:
        out[col] = out[col].astype("object")
    for idx, row in out.iterrows():
        m = row["source_metric"]
        for col in ["p05", "mean", "median", "p95"]:
            val = row[col]
            out.at[idx, col] = eur(val) if m in money else pct(val) if m in perc else number(val)
    return out.drop(columns=["source_metric"])


def main():
    st.set_page_config(page_title="AOR — итерационная модель", page_icon="📈", layout="wide")
    st.title("Art of Retention — итерационная модель Coins и фрибетов")
    st.caption("На старте у пользователей 0 фрибетов. В каждой итерации они могут выполнить миссии, получить Coins / фрибеты, использовать часть баланса и перенести остаток дальше.")

    with st.sidebar:
        st.header("1. Базовые данные букмекера")
        total_bets = st.number_input("Количество ставок за период", min_value=1, value=5_483_458, step=1_000)
        turnover = st.number_input("Оборот, €", min_value=1.0, value=40_834_160.0, step=10_000.0)
        ggr = st.number_input("GGR, €", value=4_682_582.0, step=5_000.0)
        active_users = st.number_input("Активные пользователи за период", min_value=1, value=372_326, step=100)

        st.header("2. Итерации")
        n_iterations = st.slider("Количество итераций в периоде", 1, 30, 6, 1)
        iteration_volatility = st.slider("Волатильность активности между итерациями", 0.00, 1.00, 0.15, 0.01)

        st.header("3. Воронка AOR на каждой итерации")
        target_share = st.slider("Доля пользователей, выбранных системой", 0.0, 1.0, 0.30, 0.01)
        activation_rate = st.slider("Доля пользователей, принявших миссию", 0.0, 1.0, 0.50, 0.01)
        completion_rate = st.slider("Доля пользователей, выполнивших хотя бы 1 миссию", 0.0, 1.0, 0.60, 0.01)

        st.header("4. Миссии на каждой итерации")
        avg_missions = st.slider("Среднее миссий на completed user за итерацию", 1.0, 10.0, 2.0, 0.1)
        max_missions = st.slider("Максимум миссий на completed user за итерацию", 1, 20, 5, 1)

        st.header("5. Характеристики роста")
        bet_count_uplift = st.slider("Рост количества ставок у completed users", -0.50, 2.00, 0.15, 0.01)
        avg_stake_uplift = st.slider("Рост средней ставки у completed users", -0.50, 2.00, 0.05, 0.01)
        hold_uplift = st.slider("Изменение hold / GGR margin у completed users", -0.50, 1.00, 0.00, 0.01)

        st.header("6. Coins и траты фрибетов")
        reward_budget_share = st.slider("Доля дополнительной прибыли на Coins / фрибеты", 0.0, 1.0, 0.50, 0.01)
        spend_rate = st.slider("Доля доступных фрибетов, используемых за итерацию", 0.0, 1.0, 0.50, 0.01)
        st.info("Фиксировано: 1 Coin = 1 фрибет = €1 бонусного бюджета. На старте баланс фрибетов = 0.")

        st.header("7. Настройки симуляции")
        n_simulations = st.number_input("Количество симуляций", min_value=200, value=300, step=50)
        user_bet_cv = st.slider("Коэффициент вариации ставок пользователей", 0.01, 3.00, 0.80, 0.01)
        random_seed = st.number_input("Random seed", value=42, step=1)

    avg_stake_base = safe_divide(turnover, total_bets)
    hold_base = safe_divide(ggr, turnover)
    ggr_per_bet = safe_divide(ggr, total_bets)
    bets_per_user = safe_divide(total_bets, active_users)

    st.subheader("Базовая экономика")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Средняя ставка", eur(avg_stake_base))
    b2.metric("GGR margin / hold", pct(hold_base))
    b3.metric("GGR на ставку", eur(ggr_per_bet))
    b4.metric("Ставок на active user", number(bets_per_user))
    b5.metric("Итераций", str(n_iterations))

    inp = ModelInputs(
        total_bets=int(total_bets), turnover=float(turnover), ggr=float(ggr), active_users=int(active_users),
        n_simulations=int(n_simulations), n_iterations=int(n_iterations), target_share=float(target_share),
        activation_rate=float(activation_rate), completion_rate=float(completion_rate),
        avg_missions_per_completed_user_per_iteration=float(avg_missions),
        max_missions_per_completed_user_per_iteration=int(max_missions),
        bet_count_uplift=float(bet_count_uplift), avg_stake_uplift=float(avg_stake_uplift), hold_uplift=float(hold_uplift),
        reward_budget_share=float(reward_budget_share), freebet_spend_rate_per_iteration=float(spend_rate),
        user_bet_cv=float(user_bet_cv), iteration_volatility=float(iteration_volatility), random_seed=int(random_seed),
    )
    metrics_df, iteration_df = run_simulations(inp.__dict__)

    st.subheader("Результаты симуляции")
    mean_freebets_issued = metrics_df["freebets_issued_per_active_user"].mean()
    mean_freebets_spent = metrics_df["freebets_spent_per_active_user"].mean()
    mean_final_balance = metrics_df["final_freebet_balance_per_active_user"].mean()
    mean_missions = metrics_df["avg_missions_per_completed_user"].mean()
    mean_ggr_change = metrics_df["incremental_ggr_gross"].mean()
    median_ggr_change_pct = metrics_df["gross_uplift_pct"].median()
    mean_spent_cost = metrics_df["program_cost_spent"].mean()
    mean_net_after_spent = metrics_df["net_ggr_after_spent"].mean()
    median_roi_after_spent = metrics_df["roi_after_spent"].median()
    mean_conservative_net = metrics_df["conservative_net_ggr"].mean()
    limit_share = metrics_df["reward_limit_compliance_share"].mean()
    break_even_after_spent = metrics_df["break_even_after_spent_share"].mean()

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Фрибеты выданы / active user", number(mean_freebets_issued))
    r1c2.metric("Фрибеты использованы / active user", number(mean_freebets_spent))
    r1c3.metric("Фрибеты на конец / active user", number(mean_final_balance))
    r1c4.metric("Миссий на completed user", number(mean_missions))

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Изменение GGR", eur(mean_ggr_change), delta=pct(median_ggr_change_pct))
    r2c2.metric("Стоимость использованных фрибетов", eur(mean_spent_cost))
    r2c3.metric("Net GGR после трат", eur(mean_net_after_spent))
    r2c4.metric("ROI после трат", pct(median_roi_after_spent))

    r3c1, r3c2, r3c3 = st.columns(3)
    r3c1.metric("Консервативный net GGR", eur(mean_conservative_net))
    r3c2.metric("Лимит Coins соблюдён", pct(limit_share))
    r3c3.metric("Break-even итерации после трат", pct(break_even_after_spent))

    st.markdown("""
**Итерационная логика.** На старте `Freebet Balance = 0`. В каждой итерации модель считает incremental gross GGR, выдаёт Coins / фрибеты в пределах созданной дополнительной прибыли, добавляет их к балансу, списывает использованную часть и переносит остаток в следующую итерацию.

```text
Coins / Freebets Issued Cost <= Incremental Gross GGR of Iteration
```
""")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Итерации", "Фрибеты", "Распределения", "Сводная таблица", "Данные"])

    with tab1:
        iter_summary = iteration_df.groupby("iteration").agg(
            baseline_ggr=("baseline_ggr", "median"),
            aor_ggr_gross=("aor_ggr_gross", "median"),
            net_ggr_after_spent=("net_ggr_after_spent", "median"),
            completed_missions=("completed_missions", "median"),
        ).reset_index()
        fig_ggr = go.Figure()
        fig_ggr.add_trace(go.Scatter(x=iter_summary["iteration"], y=iter_summary["baseline_ggr"], mode="lines+markers", name="Baseline GGR"))
        fig_ggr.add_trace(go.Scatter(x=iter_summary["iteration"], y=iter_summary["aor_ggr_gross"], mode="lines+markers", name="AOR gross GGR"))
        fig_ggr.add_trace(go.Scatter(x=iter_summary["iteration"], y=iter_summary["net_ggr_after_spent"], mode="lines+markers", name="Net GGR after spent freebets"))
        fig_ggr.update_layout(title="Медианный GGR по итерациям", xaxis_title="Итерация", yaxis_title="GGR, €", hovermode="x unified")
        st.plotly_chart(fig_ggr, use_container_width=True)
        fig_missions = px.bar(iter_summary, x="iteration", y="completed_missions", title="Медианное количество выполненных миссий по итерациям")
        fig_missions.update_layout(xaxis_title="Итерация", yaxis_title="Миссии")
        st.plotly_chart(fig_missions, use_container_width=True)

    with tab2:
        fb_summary = iteration_df.groupby("iteration").agg(
            freebet_balance_start=("freebet_balance_start", "median"),
            freebets_issued=("freebets_issued", "median"),
            freebets_spent=("freebets_spent", "median"),
            freebet_balance_end=("freebet_balance_end", "median"),
        ).reset_index()
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(x=fb_summary["iteration"], y=fb_summary["freebets_issued"], mode="lines+markers", name="Выдано фрибетов"))
        fig_flow.add_trace(go.Scatter(x=fb_summary["iteration"], y=fb_summary["freebets_spent"], mode="lines+markers", name="Использовано фрибетов"))
        fig_flow.update_layout(title="Выдача и использование фрибетов по итерациям", xaxis_title="Итерация", yaxis_title="Фрибеты", hovermode="x unified")
        st.plotly_chart(fig_flow, use_container_width=True)
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(x=fb_summary["iteration"], y=fb_summary["freebet_balance_start"], mode="lines+markers", name="Баланс на начало"))
        fig_balance.add_trace(go.Scatter(x=fb_summary["iteration"], y=fb_summary["freebet_balance_end"], mode="lines+markers", name="Баланс на конец"))
        fig_balance.update_layout(title="Баланс фрибетов по итерациям", xaxis_title="Итерация", yaxis_title="Фрибеты", hovermode="x unified")
        st.plotly_chart(fig_balance, use_container_width=True)

    with tab3:
        d1, d2 = st.columns(2)
        with d1:
            fig_roi = px.histogram(metrics_df, x="roi_after_spent", nbins=40, title="Распределение ROI по использованным фрибетам")
            fig_roi.update_layout(xaxis_title="ROI", yaxis_title="Симуляции")
            st.plotly_chart(fig_roi, use_container_width=True)
        with d2:
            fig_issued = px.histogram(metrics_df, x="freebets_issued_per_active_user", nbins=40, title="Распределение выданных фрибетов на active user")
            fig_issued.update_layout(xaxis_title="Фрибеты выданы / active user", yaxis_title="Симуляции")
            st.plotly_chart(fig_issued, use_container_width=True)
        d3, d4 = st.columns(2)
        with d3:
            fig_spent = px.histogram(metrics_df, x="freebets_spent_per_active_user", nbins=40, title="Распределение использованных фрибетов на active user")
            fig_spent.update_layout(xaxis_title="Фрибеты использованы / active user", yaxis_title="Симуляции")
            st.plotly_chart(fig_spent, use_container_width=True)
        with d4:
            fig_balance = px.histogram(metrics_df, x="final_freebet_balance_per_active_user", nbins=40, title="Распределение остатка фрибетов на active user")
            fig_balance.update_layout(xaxis_title="Фрибеты на конец / active user", yaxis_title="Симуляции")
            st.plotly_chart(fig_balance, use_container_width=True)

    with tab4:
        summary_df = build_summary(metrics_df)
        st.dataframe(format_summary(summary_df), use_container_width=True)
        st.download_button("Скачать сводную таблицу CSV", data=summary_df.to_csv(index=False).encode("utf-8"), file_name="aor_iterations_summary_ru.csv", mime="text/csv")

    with tab5:
        st.markdown("**Итоги по симуляциям**")
        st.dataframe(metrics_df, use_container_width=True)
        st.download_button("Скачать итоги симуляций CSV", data=metrics_df.to_csv(index=False).encode("utf-8"), file_name="aor_simulation_totals_ru.csv", mime="text/csv")
        st.markdown("**Данные по итерациям**")
        st.dataframe(iteration_df, use_container_width=True)
        st.download_button("Скачать данные по итерациям CSV", data=iteration_df.to_csv(index=False).encode("utf-8"), file_name="aor_iteration_results_ru.csv", mime="text/csv")

    st.subheader("Логика модели")
    st.code("""
На старте:
freebet_balance_0 = 0

Для каждой итерации k:
baseline_ggr_k = total_ggr * iteration_weight_k
incremental_ggr_gross_k = aor_ggr_gross_k - baseline_ggr_k
max_reward_budget_k = max(0, incremental_ggr_gross_k)
planned_reward_budget_k = max_reward_budget_k * reward_budget_share
coins_issued_k = floor(planned_reward_budget_k / EUR1)
freebets_issued_k = coins_issued_k
freebet_balance_after_issue_k = freebet_balance_start_k + freebets_issued_k
freebets_spent_k = freebet_balance_after_issue_k * freebet_spend_rate_per_iteration
freebet_balance_end_k = freebet_balance_after_issue_k - freebets_spent_k
freebet_balance_start_{k+1} = freebet_balance_end_k
program_cost_spent_k = freebets_spent_k * EUR1
program_cost_issued_k = freebets_issued_k * EUR1
net_ggr_after_spent_k = incremental_ggr_gross_k - program_cost_spent_k
conservative_net_ggr_k = incremental_ggr_gross_k - program_cost_issued_k
    """, language="text")


if __name__ == "__main__":
    main()
