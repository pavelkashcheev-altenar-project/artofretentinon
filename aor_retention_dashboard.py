import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DAYS = 30


@dataclass
class ModelInputs:
    total_bets: int
    turnover: float
    ggr: float
    active_users: int
    n_simulations: int
    target_share: float
    activation_rate: float
    completion_rate: float
    bet_count_uplift: float
    avg_stake_uplift: float
    hold_uplift: float
    coins_per_completed_mission: float
    missions_per_completed_user: float
    freebet_value_eur: float
    redemption_rate: float
    freebet_cost_factor: float
    user_bet_cv: float
    daily_volatility: float
    random_seed: int


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return default if b == 0 else a / b


def pct(x: float) -> str:
    if pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"{x * 100:,.1f}%"


def eur(x: float) -> str:
    if pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"€{x:,.0f}"


def num(x: float) -> str:
    if pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"{x:,.2f}"


def generate_gaussian_user_weights(
    rng: np.random.Generator,
    active_users: int,
    user_bet_cv: float,
) -> np.ndarray:
    """
    Generates a truncated Gaussian distribution of relative betting intensity by user.
    Mean is normalized to 1 after clipping, so total simulated baseline bets can be
    rescaled to the bookmaker's actual total bet count.
    """
    raw = rng.normal(loc=1.0, scale=user_bet_cv, size=active_users)
    weights = np.clip(raw, 0.01, None)
    return weights / weights.mean()


def generate_daily_weights(
    rng: np.random.Generator,
    daily_volatility: float,
    days: int = DAYS,
) -> np.ndarray:
    """
    Daily weights add calendar-day noise while preserving the 30-day total.
    """
    raw = rng.normal(loc=1.0, scale=daily_volatility, size=days)
    weights = np.clip(raw, 0.05, None)
    return weights / weights.sum()


def run_single_simulation(
    rng: np.random.Generator,
    inputs: ModelInputs,
    sim_id: int,
) -> Tuple[Dict, pd.DataFrame]:
    base_avg_stake = safe_divide(inputs.turnover, inputs.total_bets)
    base_hold = safe_divide(inputs.ggr, inputs.turnover)

    user_weights = generate_gaussian_user_weights(
        rng=rng,
        active_users=inputs.active_users,
        user_bet_cv=inputs.user_bet_cv,
    )

    daily_weights = generate_daily_weights(
        rng=rng,
        daily_volatility=inputs.daily_volatility,
        days=DAYS,
    )

    # Expected baseline bets per user over 30 days.
    user_bets_30d = (inputs.total_bets / inputs.active_users) * user_weights

    # Baseline daily user bets. This preserves total bet count in expectation and,
    # after rescaling, exactly matches the input total_bets at the aggregate level.
    baseline_user_day_bets = np.outer(user_bets_30d, daily_weights)
    scale_to_total = safe_divide(inputs.total_bets, baseline_user_day_bets.sum(), default=1.0)
    baseline_user_day_bets = baseline_user_day_bets * scale_to_total

    # AOR funnel.
    targeted = rng.random(inputs.active_users) < inputs.target_share
    activated = targeted & (rng.random(inputs.active_users) < inputs.activation_rate)
    completed = activated & (rng.random(inputs.active_users) < inputs.completion_rate)

    completed_multiplier_bets = np.where(completed, 1.0 + inputs.bet_count_uplift, 1.0)
    completed_multiplier_stake = np.where(completed, 1.0 + inputs.avg_stake_uplift, 1.0)
    completed_multiplier_hold = np.where(completed, 1.0 + inputs.hold_uplift, 1.0)

    aor_user_day_bets = baseline_user_day_bets * completed_multiplier_bets[:, None]

    baseline_user_day_turnover = baseline_user_day_bets * base_avg_stake
    aor_user_day_turnover = aor_user_day_bets * (base_avg_stake * completed_multiplier_stake[:, None])

    baseline_user_day_ggr = baseline_user_day_turnover * base_hold
    aor_user_day_ggr = aor_user_day_turnover * (base_hold * completed_multiplier_hold[:, None])

    baseline_bets = baseline_user_day_bets.sum()
    aor_bets = aor_user_day_bets.sum()
    baseline_turnover = baseline_user_day_turnover.sum()
    aor_turnover = aor_user_day_turnover.sum()
    baseline_ggr = baseline_user_day_ggr.sum()
    aor_ggr_gross = aor_user_day_ggr.sum()

    completed_users = int(completed.sum())
    targeted_users = int(targeted.sum())
    activated_users = int(activated.sum())

    # Course: 1 Coin = 1 freebet.
    total_freebets = (
        completed_users
        * inputs.missions_per_completed_user
        * inputs.coins_per_completed_mission
    )

    freebets_per_active_user = safe_divide(total_freebets, inputs.active_users)
    freebets_per_targeted_user = safe_divide(total_freebets, targeted_users)
    freebets_per_completed_user = safe_divide(total_freebets, completed_users)

    program_cost = (
        total_freebets
        * inputs.freebet_value_eur
        * inputs.redemption_rate
        * inputs.freebet_cost_factor
    )

    incremental_ggr_gross = aor_ggr_gross - baseline_ggr
    incremental_ggr_net = incremental_ggr_gross - program_cost
    roi = safe_divide(incremental_ggr_net, program_cost, default=np.nan)
    gross_roi = safe_divide(incremental_ggr_gross, program_cost, default=np.nan)
    net_uplift_pct = safe_divide(incremental_ggr_net, baseline_ggr, default=np.nan)
    gross_uplift_pct = safe_divide(incremental_ggr_gross, baseline_ggr, default=np.nan)

    metrics = {
        "simulation": sim_id,
        "targeted_users": targeted_users,
        "activated_users": activated_users,
        "completed_users": completed_users,
        "baseline_bets": baseline_bets,
        "aor_bets": aor_bets,
        "incremental_bets": aor_bets - baseline_bets,
        "baseline_turnover": baseline_turnover,
        "aor_turnover": aor_turnover,
        "incremental_turnover": aor_turnover - baseline_turnover,
        "baseline_ggr": baseline_ggr,
        "aor_ggr_gross": aor_ggr_gross,
        "incremental_ggr_gross": incremental_ggr_gross,
        "program_cost": program_cost,
        "incremental_ggr_net": incremental_ggr_net,
        "roi": roi,
        "gross_roi": gross_roi,
        "net_uplift_pct": net_uplift_pct,
        "gross_uplift_pct": gross_uplift_pct,
        "total_freebets": total_freebets,
        "freebets_per_active_user": freebets_per_active_user,
        "freebets_per_targeted_user": freebets_per_targeted_user,
        "freebets_per_completed_user": freebets_per_completed_user,
        "break_even": incremental_ggr_net >= 0,
    }

    daily = pd.DataFrame(
        {
            "simulation": sim_id,
            "day": np.arange(1, DAYS + 1),
            "baseline_bets": baseline_user_day_bets.sum(axis=0),
            "aor_bets": aor_user_day_bets.sum(axis=0),
            "baseline_turnover": baseline_user_day_turnover.sum(axis=0),
            "aor_turnover": aor_user_day_turnover.sum(axis=0),
            "baseline_ggr": baseline_user_day_ggr.sum(axis=0),
            "aor_ggr_gross": aor_user_day_ggr.sum(axis=0),
        }
    )
    daily["incremental_ggr_gross"] = daily["aor_ggr_gross"] - daily["baseline_ggr"]
    daily["incremental_bets"] = daily["aor_bets"] - daily["baseline_bets"]

    return metrics, daily


@st.cache_data(show_spinner=False)
def run_simulations(inputs_dict: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inputs = ModelInputs(**inputs_dict)
    rng = np.random.default_rng(inputs.random_seed)

    metric_rows = []
    daily_rows = []

    for sim_id in range(1, inputs.n_simulations + 1):
        metrics, daily = run_single_simulation(rng, inputs, sim_id)
        metric_rows.append(metrics)
        daily_rows.append(daily)

    metrics_df = pd.DataFrame(metric_rows)
    daily_df = pd.concat(daily_rows, ignore_index=True)

    return metrics_df, daily_df


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "completed_users",
        "incremental_bets",
        "incremental_turnover",
        "incremental_ggr_gross",
        "program_cost",
        "incremental_ggr_net",
        "roi",
        "net_uplift_pct",
        "total_freebets",
        "freebets_per_active_user",
    ]

    rows = []
    for key in keys:
        series = metrics_df[key].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "metric": key,
                "p05": series.quantile(0.05),
                "mean": series.mean(),
                "median": series.median(),
                "p95": series.quantile(0.95),
            }
        )

    return pd.DataFrame(rows)


def add_metric_cards(metrics_df: pd.DataFrame):
    avg_freebets = metrics_df["freebets_per_active_user"].mean()
    median_roi = metrics_df["roi"].median()
    mean_net = metrics_df["incremental_ggr_net"].mean()
    positive_share = metrics_df["break_even"].mean()
    mean_cost = metrics_df["program_cost"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Freebets / active user", f"{avg_freebets:,.2f}")
    c2.metric("Median ROI", pct(median_roi))
    c3.metric("Mean net GGR uplift", eur(mean_net))
    c4.metric("Profitable simulations", pct(positive_share))
    c5.metric("Mean program cost", eur(mean_cost))


def main():
    st.set_page_config(
        page_title="AOR Simulation Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("Art of Retention — 30-Day Simulation Dashboard")
    st.caption(
        "Monte Carlo model for missions, Coins, freebets and incremental GGR. "
        "The model runs at least 200 simulations and distributes bets across users with a truncated Gaussian distribution."
    )

    with st.sidebar:
        st.header("1. Bookmaker baseline")
        total_bets = st.number_input(
            "Number of bets in the period",
            min_value=1,
            value=100_000,
            step=1_000,
        )
        turnover = st.number_input(
            "Turnover, €",
            min_value=1.0,
            value=2_000_000.0,
            step=10_000.0,
        )
        ggr = st.number_input(
            "GGR, €",
            value=160_000.0,
            step=5_000.0,
        )
        active_users = st.number_input(
            "Active users in the period",
            min_value=1,
            value=10_000,
            step=100,
            help="Required for distributing bets across users. Without this, user-level simulation is not mathematically defined.",
        )

        st.header("2. AOR funnel")
        target_share = st.slider("Targeted users share", 0.0, 1.0, 0.30, 0.01)
        activation_rate = st.slider("Mission activation rate", 0.0, 1.0, 0.50, 0.01)
        completion_rate = st.slider("Mission completion rate", 0.0, 1.0, 0.60, 0.01)

        st.header("3. Growth assumptions")
        bet_count_uplift = st.slider("Bet count uplift for completed users", -0.50, 2.00, 0.15, 0.01)
        avg_stake_uplift = st.slider("Average stake uplift for completed users", -0.50, 2.00, 0.05, 0.01)
        hold_uplift = st.slider("Hold / GGR margin uplift for completed users", -0.50, 1.00, 0.00, 0.01)

        st.header("4. Coins and freebets")
        coins_per_completed_mission = st.number_input(
            "Coins per completed mission",
            min_value=0.0,
            value=2.0,
            step=1.0,
            help="In this model: 1 Coin = 1 freebet.",
        )
        missions_per_completed_user = st.number_input(
            "Missions per completed user in 30 days",
            min_value=0.0,
            value=1.0,
            step=0.25,
        )
        freebet_value_eur = st.number_input(
            "Value of 1 freebet, €",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )
        redemption_rate = st.slider("Freebet redemption rate", 0.0, 1.0, 0.80, 0.01)
        freebet_cost_factor = st.slider(
            "Freebet economic cost factor",
            0.0,
            1.5,
            0.75,
            0.01,
            help="Economic cost of the freebet as a share of face value. Example: €2 freebet × 75% = €1.50 expected cost before redemption.",
        )

        st.header("5. Simulation settings")
        n_simulations = st.number_input(
            "Number of simulations",
            min_value=200,
            value=300,
            step=50,
        )
        user_bet_cv = st.slider(
            "Gaussian user bet distribution CV",
            0.01,
            3.00,
            0.80,
            0.01,
            help="Higher value means more uneven betting activity between users.",
        )
        daily_volatility = st.slider(
            "Daily activity volatility",
            0.00,
            1.00,
            0.15,
            0.01,
        )
        random_seed = st.number_input("Random seed", value=42, step=1)

    base_avg_stake = safe_divide(turnover, total_bets)
    base_hold = safe_divide(ggr, turnover)
    base_ggr_per_bet = safe_divide(ggr, total_bets)
    bets_per_user = safe_divide(total_bets, active_users)

    st.subheader("Baseline economics")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Average stake", eur(base_avg_stake))
    b2.metric("GGR margin / hold", pct(base_hold))
    b3.metric("GGR per bet", eur(base_ggr_per_bet))
    b4.metric("Bets per active user", f"{bets_per_user:,.2f}")

    inputs = ModelInputs(
        total_bets=int(total_bets),
        turnover=float(turnover),
        ggr=float(ggr),
        active_users=int(active_users),
        n_simulations=int(n_simulations),
        target_share=float(target_share),
        activation_rate=float(activation_rate),
        completion_rate=float(completion_rate),
        bet_count_uplift=float(bet_count_uplift),
        avg_stake_uplift=float(avg_stake_uplift),
        hold_uplift=float(hold_uplift),
        coins_per_completed_mission=float(coins_per_completed_mission),
        missions_per_completed_user=float(missions_per_completed_user),
        freebet_value_eur=float(freebet_value_eur),
        redemption_rate=float(redemption_rate),
        freebet_cost_factor=float(freebet_cost_factor),
        user_bet_cv=float(user_bet_cv),
        daily_volatility=float(daily_volatility),
        random_seed=int(random_seed),
    )

    metrics_df, daily_df = run_simulations(inputs.__dict__)

    st.subheader("Simulation results")
    add_metric_cards(metrics_df)

    st.markdown(
        """
        **Definitions.**  
        **Freebets / active user** = total freebets generated by Coins divided by active users.  
        **ROI** = `(incremental gross GGR − freebet cost) / freebet cost`.  
        **Profitable simulations** = share of simulations where net incremental GGR is positive.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "30-day curves",
            "Distributions",
            "Summary table",
            "Raw simulation data",
        ]
    )

    with tab1:
        daily_summary = (
            daily_df.groupby("day")
            .agg(
                baseline_ggr_median=("baseline_ggr", "median"),
                aor_ggr_median=("aor_ggr_gross", "median"),
                incremental_ggr_median=("incremental_ggr_gross", "median"),
                baseline_bets_median=("baseline_bets", "median"),
                aor_bets_median=("aor_bets", "median"),
            )
            .reset_index()
        )

        fig_ggr = go.Figure()
        fig_ggr.add_trace(
            go.Scatter(
                x=daily_summary["day"],
                y=daily_summary["baseline_ggr_median"],
                mode="lines",
                name="Baseline GGR",
            )
        )
        fig_ggr.add_trace(
            go.Scatter(
                x=daily_summary["day"],
                y=daily_summary["aor_ggr_median"],
                mode="lines",
                name="AOR gross GGR",
            )
        )
        fig_ggr.update_layout(
            title="Median daily GGR: baseline vs AOR",
            xaxis_title="Day",
            yaxis_title="GGR, €",
            hovermode="x unified",
        )
        st.plotly_chart(fig_ggr, use_container_width=True)

        fig_bets = go.Figure()
        fig_bets.add_trace(
            go.Scatter(
                x=daily_summary["day"],
                y=daily_summary["baseline_bets_median"],
                mode="lines",
                name="Baseline bets",
            )
        )
        fig_bets.add_trace(
            go.Scatter(
                x=daily_summary["day"],
                y=daily_summary["aor_bets_median"],
                mode="lines",
                name="AOR bets",
            )
        )
        fig_bets.update_layout(
            title="Median daily bets: baseline vs AOR",
            xaxis_title="Day",
            yaxis_title="Bets",
            hovermode="x unified",
        )
        st.plotly_chart(fig_bets, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            fig_roi = px.histogram(
                metrics_df,
                x="roi",
                nbins=40,
                title="Distribution of program ROI",
            )
            fig_roi.update_layout(xaxis_title="ROI", yaxis_title="Simulations")
            st.plotly_chart(fig_roi, use_container_width=True)

        with c2:
            fig_freebets = px.histogram(
                metrics_df,
                x="freebets_per_active_user",
                nbins=40,
                title="Distribution of freebets per active user",
            )
            fig_freebets.update_layout(
                xaxis_title="Freebets per active user",
                yaxis_title="Simulations",
            )
            st.plotly_chart(fig_freebets, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig_net = px.histogram(
                metrics_df,
                x="incremental_ggr_net",
                nbins=40,
                title="Distribution of net incremental GGR",
            )
            fig_net.update_layout(xaxis_title="Net incremental GGR, €", yaxis_title="Simulations")
            st.plotly_chart(fig_net, use_container_width=True)

        with c4:
            fig_cost = px.histogram(
                metrics_df,
                x="program_cost",
                nbins=40,
                title="Distribution of program cost",
            )
            fig_cost.update_layout(xaxis_title="Program cost, €", yaxis_title="Simulations")
            st.plotly_chart(fig_cost, use_container_width=True)

    with tab3:
        summary_df = build_summary(metrics_df)

        formatted = summary_df.copy()
        money_metrics = {
            "incremental_turnover",
            "incremental_ggr_gross",
            "program_cost",
            "incremental_ggr_net",
        }
        pct_metrics = {"roi", "net_uplift_pct"}

        # Pandas on Streamlit Cloud may reject assigning formatted strings
        # into float columns. Build display columns as object/string columns first.
        display_cols = ["p05", "mean", "median", "p95"]
        for col in display_cols:
            formatted[col] = formatted[col].astype("object")

        for idx, row in formatted.iterrows():
            metric = row["metric"]
            for col in display_cols:
                value = row[col]
                if metric in money_metrics:
                    formatted.at[idx, col] = eur(value)
                elif metric in pct_metrics:
                    formatted.at[idx, col] = pct(value)
                else:
                    formatted.at[idx, col] = f"{value:,.2f}"

        st.dataframe(formatted, use_container_width=True)

        st.download_button(
            "Download summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="aor_simulation_summary.csv",
            mime="text/csv",
        )

    with tab4:
        st.dataframe(metrics_df, use_container_width=True)

        st.download_button(
            "Download raw simulation CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="aor_raw_simulations.csv",
            mime="text/csv",
        )

    st.subheader("Model logic")
    st.code(
        """
Baseline:
average_stake = turnover / bets
hold = GGR / turnover
GGR_per_bet = GGR / bets

User-level simulation:
user_bet_weight ~ truncated Gaussian(mean=1, std=user_bet_cv)
baseline_user_bets_30d = total_bets / active_users * user_bet_weight

AOR funnel:
completed_user = targeted * activated * completed

Growth:
AOR bets = baseline bets * (1 + bet_count_uplift for completed users)
AOR stake = baseline average stake * (1 + avg_stake_uplift for completed users)
AOR hold = baseline hold * (1 + hold_uplift for completed users)

Coins:
1 Coin = 1 freebet
total_freebets = completed_users * missions_per_completed_user * coins_per_completed_mission
program_cost = total_freebets * freebet_value_eur * redemption_rate * freebet_cost_factor

Efficiency:
incremental_ggr_gross = AOR gross GGR - baseline GGR
incremental_ggr_net = incremental_ggr_gross - program_cost
ROI = incremental_ggr_net / program_cost
        """,
        language="text",
    )


if __name__ == "__main__":
    main()
