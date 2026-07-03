import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Tuple


DAYS = 30
COIN_COST_EUR = 1.0  # Текущая версия: 1 Coin = 1 фрибет = €1 бонусного бюджета


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

    avg_missions_per_completed_user: float
    max_missions_per_completed_user: int

    bet_count_uplift: float
    avg_stake_uplift: float
    hold_uplift: float

    reward_budget_share: float

    initial_freebets_per_active_user: float
    daily_freebet_spend_rate: float

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


def number(x: float, decimals: int = 2) -> str:
    if pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"{x:,.{decimals}f}"


def generate_gaussian_user_weights(
    rng: np.random.Generator,
    active_users: int,
    user_bet_cv: float,
) -> np.ndarray:
    """
    Усечённое нормальное распределение интенсивности ставок по пользователям.
    Среднее нормализуется к 1, чтобы сумма ставок совпадала с total_bets.
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
    Дневная волатильность активности за 30 календарных дней.
    Сумма весов равна 1.
    """
    raw = rng.normal(loc=1.0, scale=daily_volatility, size=days)
    weights = np.clip(raw, 0.05, None)
    return weights / weights.sum()


def simulate_missions(
    rng: np.random.Generator,
    completed_users: int,
    avg_missions_per_completed_user: float,
    max_missions_per_completed_user: int,
    daily_weights: np.ndarray,
) -> Tuple[int, np.ndarray, float]:
    """
    Один пользователь может выполнить больше одной миссии в месяц.

    completed_users означает пользователей, которые выполнили хотя бы одну миссию.
    Для таких пользователей количество миссий моделируется как:
    1 + Poisson(avg_missions_per_completed_user - 1), с ограничением max_missions_per_completed_user.
    """
    if completed_users <= 0:
        return 0, np.zeros(DAYS), 0.0

    poisson_lambda = max(avg_missions_per_completed_user - 1.0, 0.0)
    mission_counts = 1 + rng.poisson(lam=poisson_lambda, size=completed_users)
    mission_counts = np.clip(mission_counts, 1, max_missions_per_completed_user)

    completed_missions_total = int(mission_counts.sum())
    daily_completed_missions = rng.multinomial(completed_missions_total, daily_weights)

    actual_avg_missions = safe_divide(completed_missions_total, completed_users)

    return completed_missions_total, daily_completed_missions.astype(float), actual_avg_missions


def simulate_freebet_balances(
    daily_aor_freebets_issued: np.ndarray,
    initial_freebets_total: float,
    daily_freebet_spend_rate: float,
) -> pd.DataFrame:
    """
    Пользователи могут тратить как стартовые фрибеты, так и фрибеты, полученные во время симуляции.

    Упрощение v1:
    - стартовый баланс и AOR-баланс учитываются отдельно;
    - каждый день тратится заданная доля доступного баланса;
    - новые AOR-фрибеты становятся доступными в день выдачи.
    """
    initial_balance = float(initial_freebets_total)
    aor_balance = 0.0

    rows = []

    for day_idx, issued in enumerate(daily_aor_freebets_issued, start=1):
        aor_balance += float(issued)

        initial_spent = initial_balance * daily_freebet_spend_rate
        aor_spent = aor_balance * daily_freebet_spend_rate

        initial_balance -= initial_spent
        aor_balance -= aor_spent

        rows.append(
            {
                "day": day_idx,
                "initial_freebets_spent": initial_spent,
                "aor_freebets_issued": float(issued),
                "aor_freebets_spent": aor_spent,
                "total_freebets_spent": initial_spent + aor_spent,
                "initial_freebet_balance_end": initial_balance,
                "aor_freebet_balance_end": aor_balance,
                "total_freebet_balance_end": initial_balance + aor_balance,
            }
        )

    return pd.DataFrame(rows)


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

    user_bets_30d = (inputs.total_bets / inputs.active_users) * user_weights

    targeted = rng.random(inputs.active_users) < inputs.target_share
    activated = targeted & (rng.random(inputs.active_users) < inputs.activation_rate)
    completed = activated & (rng.random(inputs.active_users) < inputs.completion_rate)

    targeted_users = int(targeted.sum())
    activated_users = int(activated.sum())
    completed_users = int(completed.sum())

    completed_base_bets = float(user_bets_30d[completed].sum())
    non_completed_base_bets = float(inputs.total_bets - completed_base_bets)

    completed_base_turnover = completed_base_bets * base_avg_stake
    non_completed_base_turnover = non_completed_base_bets * base_avg_stake

    bet_multiplier = 1.0 + inputs.bet_count_uplift
    stake_multiplier = 1.0 + inputs.avg_stake_uplift
    hold_multiplier = 1.0 + inputs.hold_uplift

    baseline_bets = float(inputs.total_bets)
    aor_bets = non_completed_base_bets + completed_base_bets * bet_multiplier

    baseline_turnover = float(inputs.turnover)
    aor_turnover = non_completed_base_turnover + completed_base_turnover * bet_multiplier * stake_multiplier

    baseline_ggr = float(inputs.ggr)
    aor_ggr_gross = (
        non_completed_base_turnover * base_hold
        + completed_base_turnover * bet_multiplier * stake_multiplier * base_hold * hold_multiplier
    )

    incremental_ggr_gross = aor_ggr_gross - baseline_ggr

    daily_weights = generate_daily_weights(
        rng=rng,
        daily_volatility=inputs.daily_volatility,
        days=DAYS,
    )

    completed_missions_total, daily_completed_missions, actual_avg_missions = simulate_missions(
        rng=rng,
        completed_users=completed_users,
        avg_missions_per_completed_user=inputs.avg_missions_per_completed_user,
        max_missions_per_completed_user=inputs.max_missions_per_completed_user,
        daily_weights=daily_weights,
    )

    # Главное ограничение модели:
    # максимальное количество Coins / фрибетов не может стоить больше,
    # чем дополнительная прибыль от применения программы.
    max_reward_budget = max(0.0, incremental_ggr_gross)
    planned_reward_budget = max_reward_budget * inputs.reward_budget_share

    max_coins_total = np.floor(max_reward_budget / COIN_COST_EUR)
    coins_total = np.floor(planned_reward_budget / COIN_COST_EUR)

    # Если миссий нет, выдавать Coins некуда.
    if completed_missions_total <= 0:
        coins_total = 0.0

    freebets_issued_total = coins_total  # 1 Coin = 1 фрибет
    coins_per_completed_mission = safe_divide(coins_total, completed_missions_total)
    freebets_per_completed_mission = coins_per_completed_mission

    if completed_missions_total > 0:
        daily_aor_freebets_issued = freebets_issued_total * daily_completed_missions / completed_missions_total
    else:
        daily_aor_freebets_issued = np.zeros(DAYS)

    initial_freebets_total = inputs.initial_freebets_per_active_user * inputs.active_users

    freebet_daily = simulate_freebet_balances(
        daily_aor_freebets_issued=daily_aor_freebets_issued,
        initial_freebets_total=initial_freebets_total,
        daily_freebet_spend_rate=inputs.daily_freebet_spend_rate,
    )

    initial_freebets_spent_total = float(freebet_daily["initial_freebets_spent"].sum())
    aor_freebets_spent_total = float(freebet_daily["aor_freebets_spent"].sum())
    total_freebets_spent = float(freebet_daily["total_freebets_spent"].sum())

    initial_freebets_end_balance = float(freebet_daily["initial_freebet_balance_end"].iloc[-1])
    aor_freebets_end_balance = float(freebet_daily["aor_freebet_balance_end"].iloc[-1])
    total_freebets_end_balance = float(freebet_daily["total_freebet_balance_end"].iloc[-1])

    # Стоимость программы по фактически потраченным AOR-фрибетам.
    program_cost_spent = aor_freebets_spent_total * COIN_COST_EUR

    # Консервативный взгляд: считаем весь выданный объём Coins / фрибетов как обязательство периода.
    program_cost_issued = freebets_issued_total * COIN_COST_EUR

    incremental_ggr_net_after_spent = incremental_ggr_gross - program_cost_spent
    incremental_ggr_net_conservative = incremental_ggr_gross - program_cost_issued

    roi_after_spent = safe_divide(incremental_ggr_net_after_spent, program_cost_spent, default=np.nan)
    roi_conservative = safe_divide(incremental_ggr_net_conservative, program_cost_issued, default=np.nan)

    gross_uplift_pct = safe_divide(incremental_ggr_gross, baseline_ggr, default=np.nan)
    net_uplift_pct_after_spent = safe_divide(incremental_ggr_net_after_spent, baseline_ggr, default=np.nan)
    net_uplift_pct_conservative = safe_divide(incremental_ggr_net_conservative, baseline_ggr, default=np.nan)

    baseline_daily_bets = baseline_bets * daily_weights
    aor_daily_bets = aor_bets * daily_weights

    baseline_daily_turnover = baseline_turnover * daily_weights
    aor_daily_turnover = aor_turnover * daily_weights

    baseline_daily_ggr = baseline_ggr * daily_weights
    aor_daily_ggr = aor_ggr_gross * daily_weights

    daily = pd.DataFrame(
        {
            "simulation": sim_id,
            "day": np.arange(1, DAYS + 1),
            "baseline_bets": baseline_daily_bets,
            "aor_bets": aor_daily_bets,
            "baseline_turnover": baseline_daily_turnover,
            "aor_turnover": aor_daily_turnover,
            "baseline_ggr": baseline_daily_ggr,
            "aor_ggr_gross": aor_daily_ggr,
            "completed_missions": daily_completed_missions,
        }
    )
    daily["incremental_ggr_gross"] = daily["aor_ggr_gross"] - daily["baseline_ggr"]

    daily = daily.merge(freebet_daily, on="day", how="left")
    daily["aor_freebet_cost_spent"] = daily["aor_freebets_spent"] * COIN_COST_EUR
    daily["initial_freebet_cost_spent"] = daily["initial_freebets_spent"] * COIN_COST_EUR
    daily["net_incremental_ggr_after_spent_aor_freebets"] = (
        daily["incremental_ggr_gross"] - daily["aor_freebet_cost_spent"]
    )

    metrics = {
        "simulation": sim_id,

        "targeted_users": targeted_users,
        "activated_users": activated_users,
        "completed_users": completed_users,

        "completed_missions_total": completed_missions_total,
        "actual_avg_missions_per_completed_user": actual_avg_missions,

        "baseline_bets": baseline_bets,
        "aor_bets": aor_bets,
        "incremental_bets": aor_bets - baseline_bets,

        "baseline_turnover": baseline_turnover,
        "aor_turnover": aor_turnover,
        "incremental_turnover": aor_turnover - baseline_turnover,

        "baseline_ggr": baseline_ggr,
        "aor_ggr_gross": aor_ggr_gross,
        "incremental_ggr_gross": incremental_ggr_gross,
        "gross_uplift_pct": gross_uplift_pct,

        "max_reward_budget": max_reward_budget,
        "planned_reward_budget": planned_reward_budget,

        "max_coins_total": max_coins_total,
        "coins_total": coins_total,
        "coins_per_completed_mission": coins_per_completed_mission,
        "coins_per_active_user": safe_divide(coins_total, inputs.active_users),
        "max_coins_per_active_user": safe_divide(max_coins_total, inputs.active_users),

        "aor_freebets_issued_total": freebets_issued_total,
        "aor_freebets_spent_total": aor_freebets_spent_total,
        "aor_freebets_end_balance": aor_freebets_end_balance,

        "aor_freebets_issued_per_active_user": safe_divide(freebets_issued_total, inputs.active_users),
        "aor_freebets_spent_per_active_user": safe_divide(aor_freebets_spent_total, inputs.active_users),
        "aor_freebets_end_balance_per_active_user": safe_divide(aor_freebets_end_balance, inputs.active_users),

        "freebets_per_completed_mission": freebets_per_completed_mission,

        "initial_freebets_start_total": initial_freebets_total,
        "initial_freebets_spent_total": initial_freebets_spent_total,
        "initial_freebets_end_balance": initial_freebets_end_balance,

        "total_freebets_spent": total_freebets_spent,
        "total_freebets_end_balance": total_freebets_end_balance,

        "program_cost_spent": program_cost_spent,
        "program_cost_issued": program_cost_issued,

        "incremental_ggr_net_after_spent": incremental_ggr_net_after_spent,
        "incremental_ggr_net_conservative": incremental_ggr_net_conservative,

        "roi_after_spent": roi_after_spent,
        "roi_conservative": roi_conservative,

        "net_uplift_pct_after_spent": net_uplift_pct_after_spent,
        "net_uplift_pct_conservative": net_uplift_pct_conservative,

        "reward_budget_does_not_exceed_incremental_profit": program_cost_issued <= max(0.0, incremental_ggr_gross),
        "break_even_after_spent_rewards": incremental_ggr_net_after_spent >= 0,
        "break_even_conservative": incremental_ggr_net_conservative >= 0,
    }

    return metrics, daily


@st.cache_data(show_spinner=False)
def run_simulations(inputs_dict: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inputs = ModelInputs(**inputs_dict)
    rng = np.random.default_rng(inputs.random_seed)

    metric_rows = []
    daily_frames = []

    for sim_id in range(1, inputs.n_simulations + 1):
        row, daily = run_single_simulation(rng, inputs, sim_id)
        metric_rows.append(row)
        daily_frames.append(daily)

    return pd.DataFrame(metric_rows), pd.concat(daily_frames, ignore_index=True)


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_order = [
        "completed_users",
        "completed_missions_total",
        "actual_avg_missions_per_completed_user",

        "incremental_bets",
        "incremental_turnover",
        "incremental_ggr_gross",
        "gross_uplift_pct",

        "max_reward_budget",
        "planned_reward_budget",

        "max_coins_total",
        "coins_total",
        "coins_per_completed_mission",
        "coins_per_active_user",
        "max_coins_per_active_user",

        "aor_freebets_issued_total",
        "aor_freebets_spent_total",
        "aor_freebets_end_balance",
        "aor_freebets_issued_per_active_user",
        "aor_freebets_spent_per_active_user",
        "aor_freebets_end_balance_per_active_user",

        "initial_freebets_start_total",
        "initial_freebets_spent_total",
        "initial_freebets_end_balance",

        "program_cost_spent",
        "program_cost_issued",

        "incremental_ggr_net_after_spent",
        "incremental_ggr_net_conservative",

        "roi_after_spent",
        "roi_conservative",

        "net_uplift_pct_after_spent",
        "net_uplift_pct_conservative",
    ]

    rows = []
    for metric in metric_order:
        s = metrics_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "metric": metric,
                "p05": s.quantile(0.05),
                "mean": s.mean(),
                "median": s.median(),
                "p95": s.quantile(0.95),
            }
        )

    return pd.DataFrame(rows)


def format_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    names = {
        "completed_users": "Пользователи, выполнившие хотя бы 1 миссию",
        "completed_missions_total": "Выполненные миссии всего",
        "actual_avg_missions_per_completed_user": "Миссий на completed user",

        "incremental_bets": "Дополнительные ставки",
        "incremental_turnover": "Дополнительный оборот",
        "incremental_ggr_gross": "Изменение GGR / дополнительный gross GGR",
        "gross_uplift_pct": "Изменение GGR, % к baseline",

        "max_reward_budget": "Максимальный фонд Coins / фрибетов",
        "planned_reward_budget": "Плановый фонд Coins / фрибетов",

        "max_coins_total": "Максимум Coins всего",
        "coins_total": "Расчётные Coins всего",
        "coins_per_completed_mission": "Coins на выполненную миссию",
        "coins_per_active_user": "Coins на active user",
        "max_coins_per_active_user": "Максимум Coins на active user",

        "aor_freebets_issued_total": "AOR-фрибеты выданы всего",
        "aor_freebets_spent_total": "AOR-фрибеты использованы всего",
        "aor_freebets_end_balance": "AOR-фрибеты на конец периода",
        "aor_freebets_issued_per_active_user": "AOR-фрибеты выданы на active user",
        "aor_freebets_spent_per_active_user": "AOR-фрибеты использованы на active user",
        "aor_freebets_end_balance_per_active_user": "AOR-фрибеты на конец на active user",

        "initial_freebets_start_total": "Стартовые фрибеты на начало",
        "initial_freebets_spent_total": "Стартовые фрибеты использованы",
        "initial_freebets_end_balance": "Стартовые фрибеты на конец",

        "program_cost_spent": "Стоимость использованных AOR-фрибетов",
        "program_cost_issued": "Консервативная стоимость выданных AOR-фрибетов",

        "incremental_ggr_net_after_spent": "Net GGR после использованных AOR-фрибетов",
        "incremental_ggr_net_conservative": "Консервативный Net GGR после выданных AOR-фрибетов",

        "roi_after_spent": "ROI по использованным AOR-фрибетам",
        "roi_conservative": "Консервативный ROI по выданным AOR-фрибетам",

        "net_uplift_pct_after_spent": "Net uplift после использованных AOR-фрибетов",
        "net_uplift_pct_conservative": "Консервативный net uplift после выданных AOR-фрибетов",
    }

    money_metrics = {
        "incremental_turnover",
        "incremental_ggr_gross",
        "max_reward_budget",
        "planned_reward_budget",
        "program_cost_spent",
        "program_cost_issued",
        "incremental_ggr_net_after_spent",
        "incremental_ggr_net_conservative",
    }
    pct_metrics = {
        "gross_uplift_pct",
        "roi_after_spent",
        "roi_conservative",
        "net_uplift_pct_after_spent",
        "net_uplift_pct_conservative",
    }

    formatted = summary_df.copy()
    formatted["source_metric"] = formatted["metric"]
    formatted["metric"] = formatted["metric"].map(names).fillna(formatted["metric"])

    for col in ["p05", "mean", "median", "p95"]:
        formatted[col] = formatted[col].astype("object")

    for idx, row in formatted.iterrows():
        metric = row["source_metric"]
        for col in ["p05", "mean", "median", "p95"]:
            value = row[col]
            if metric in money_metrics:
                formatted.at[idx, col] = eur(value)
            elif metric in pct_metrics:
                formatted.at[idx, col] = pct(value)
            else:
                formatted.at[idx, col] = number(value)

    return formatted.drop(columns=["source_metric"])


def main():
    st.set_page_config(
        page_title="AOR — Coins, фрибеты и миссии",
        page_icon="📈",
        layout="wide",
    )

    st.title("Art of Retention — симуляционная модель с миссиями, Coins и фрибетами")
    st.caption(
        "Модель учитывает стартовые фрибеты, траты фрибетов в течение 30 дней "
        "и возможность выполнить больше одной миссии на пользователя в месяц."
    )

    with st.sidebar:
        st.header("1. Базовые данные букмекера")
        total_bets = st.number_input(
            "Количество ставок за период",
            min_value=1,
            value=5_483_458,
            step=1_000,
        )
        turnover = st.number_input(
            "Оборот, €",
            min_value=1.0,
            value=40_834_160.0,
            step=10_000.0,
        )
        ggr = st.number_input(
            "GGR, €",
            value=4_682_582.0,
            step=5_000.0,
        )
        active_users = st.number_input(
            "Активные пользователи за период",
            min_value=1,
            value=372_326,
            step=100,
            help="Нужно для распределения ставок между пользователями.",
        )

        st.header("2. Воронка AOR")
        target_share = st.slider("Доля пользователей, выбранных системой", 0.0, 1.0, 0.30, 0.01)
        activation_rate = st.slider("Доля пользователей, принявших миссию", 0.0, 1.0, 0.50, 0.01)
        completion_rate = st.slider("Доля пользователей, выполнивших хотя бы 1 миссию", 0.0, 1.0, 0.60, 0.01)

        st.header("3. Миссии")
        avg_missions_per_completed_user = st.slider(
            "Среднее количество миссий на completed user в месяц",
            1.0,
            10.0,
            2.0,
            0.1,
            help="Пользователь, который выполнил миссию, может выполнить больше одной миссии за 30 дней.",
        )
        max_missions_per_completed_user = st.slider(
            "Максимум миссий на completed user в месяц",
            1,
            20,
            5,
            1,
        )

        st.header("4. Характеристики роста")
        bet_count_uplift = st.slider("Рост количества ставок у выполнивших миссию", -0.50, 2.00, 0.15, 0.01)
        avg_stake_uplift = st.slider("Рост средней ставки у выполнивших миссию", -0.50, 2.00, 0.05, 0.01)
        hold_uplift = st.slider("Изменение hold / GGR margin у выполнивших миссию", -0.50, 1.00, 0.00, 0.01)

        st.header("5. Расчёт Coins")
        reward_budget_share = st.slider(
            "Доля дополнительной прибыли на Coins / фрибеты",
            0.0,
            1.0,
            0.50,
            0.01,
            help=(
                "Например, 50% означает, что на Coins можно направить половину дополнительного gross GGR. "
                "Даже при 100% фонд не превысит дополнительную прибыль."
            ),
        )
        st.info("Фиксировано в этой версии: 1 Coin = 1 фрибет = €1 бонусного бюджета.")

        st.header("6. Стартовые фрибеты и траты")
        initial_freebets_per_active_user = st.number_input(
            "Стартовые фрибеты на active user",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Фрибеты, которые уже есть у пользователей в начале симуляции. Это не AOR-выдача, а стартовый баланс.",
        )
        daily_freebet_spend_rate = st.slider(
            "Доля доступных фрибетов, используемых в день",
            0.0,
            1.0,
            0.05,
            0.01,
            help="Применяется отдельно к стартовому балансу и к AOR-фрибетам, которые появляются в течение симуляции.",
        )

        st.header("7. Настройки симуляции")
        n_simulations = st.number_input(
            "Количество симуляций",
            min_value=200,
            value=300,
            step=50,
        )
        user_bet_cv = st.slider(
            "Коэффициент вариации ставок пользователей",
            0.01,
            3.00,
            0.80,
            0.01,
            help="Чем выше значение, тем сильнее различается активность пользователей.",
        )
        daily_volatility = st.slider("Дневная волатильность активности", 0.00, 1.00, 0.15, 0.01)
        random_seed = st.number_input("Random seed", value=42, step=1)

    base_avg_stake = safe_divide(turnover, total_bets)
    base_hold = safe_divide(ggr, turnover)
    base_ggr_per_bet = safe_divide(ggr, total_bets)
    bets_per_user = safe_divide(total_bets, active_users)

    st.subheader("Базовая экономика")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Средняя ставка", eur(base_avg_stake))
    b2.metric("GGR margin / hold", pct(base_hold))
    b3.metric("GGR на одну ставку", eur(base_ggr_per_bet))
    b4.metric("Ставок на active user", number(bets_per_user))

    inputs = ModelInputs(
        total_bets=int(total_bets),
        turnover=float(turnover),
        ggr=float(ggr),
        active_users=int(active_users),
        n_simulations=int(n_simulations),

        target_share=float(target_share),
        activation_rate=float(activation_rate),
        completion_rate=float(completion_rate),

        avg_missions_per_completed_user=float(avg_missions_per_completed_user),
        max_missions_per_completed_user=int(max_missions_per_completed_user),

        bet_count_uplift=float(bet_count_uplift),
        avg_stake_uplift=float(avg_stake_uplift),
        hold_uplift=float(hold_uplift),

        reward_budget_share=float(reward_budget_share),

        initial_freebets_per_active_user=float(initial_freebets_per_active_user),
        daily_freebet_spend_rate=float(daily_freebet_spend_rate),

        user_bet_cv=float(user_bet_cv),
        daily_volatility=float(daily_volatility),
        random_seed=int(random_seed),
    )

    metrics_df, daily_df = run_simulations(inputs.__dict__)

    st.subheader("Результаты симуляции")

    avg_freebets_issued = metrics_df["aor_freebets_issued_per_active_user"].mean()
    avg_freebets_spent = metrics_df["aor_freebets_spent_per_active_user"].mean()
    avg_max_coins = metrics_df["max_coins_per_active_user"].mean()
    avg_missions_actual = metrics_df["actual_avg_missions_per_completed_user"].mean()

    mean_ggr_change = metrics_df["incremental_ggr_gross"].mean()
    median_ggr_change_pct = metrics_df["gross_uplift_pct"].median()

    mean_aor_spent_cost = metrics_df["program_cost_spent"].mean()
    mean_net_after_spent = metrics_df["incremental_ggr_net_after_spent"].mean()
    median_roi_after_spent = metrics_df["roi_after_spent"].median()

    mean_conservative_net = metrics_df["incremental_ggr_net_conservative"].mean()
    mean_initial_spent = metrics_df["initial_freebets_spent_total"].mean()
    mean_end_balance = metrics_df["total_freebets_end_balance"].mean()
    constraint_share = metrics_df["reward_budget_does_not_exceed_incremental_profit"].mean()

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("AOR-фрибеты выданы / active user", number(avg_freebets_issued))
    r1c2.metric("AOR-фрибеты использованы / active user", number(avg_freebets_spent))
    r1c3.metric("Макс. Coins / active user", number(avg_max_coins))
    r1c4.metric("Миссий на completed user", number(avg_missions_actual))

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Изменение GGR", eur(mean_ggr_change), delta=pct(median_ggr_change_pct))
    r2c2.metric("Стоимость использованных AOR-фрибетов", eur(mean_aor_spent_cost))
    r2c3.metric("Net GGR после трат", eur(mean_net_after_spent))
    r2c4.metric("ROI после трат", pct(median_roi_after_spent))

    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    r3c1.metric("Консервативный net GGR", eur(mean_conservative_net))
    r3c2.metric("Стартовые фрибеты использованы", eur(mean_initial_spent))
    r3c3.metric("Баланс фрибетов на конец", number(mean_end_balance))
    r3c4.metric("Лимит Coins соблюдён", pct(constraint_share))

    st.markdown(
        """
        **Что добавлено в этой версии.**

        1. У пользователей может быть больше одной миссии в месяц.  
        Количество миссий моделируется на completed user, а затем распределяется по 30 дням.

        2. У пользователей может быть стартовый баланс фрибетов на начало симуляции.  
        Эти фрибеты учитываются отдельно от AOR-фрибетов.

        3. Пользователи могут тратить фрибеты в течение симуляции.  
        Каждый день модель списывает заданную долю доступного баланса: отдельно по стартовым фрибетам и отдельно по AOR-фрибетам.

        4. AOR Coins и AOR-фрибеты по-прежнему не вводятся вручную.  
        Они рассчитываются из дополнительного gross GGR:

        ```text
        Max Reward Budget = max(0, Incremental Gross GGR)
        Planned Reward Budget = Max Reward Budget × Reward Budget Share
        Coins = floor(Planned Reward Budget / €1)
        AOR Freebets Issued = Coins
        ```

        Стоимость выданных AOR Coins / фрибетов не может превышать дополнительную прибыль от программы.
        """
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Динамика за 30 дней",
            "Фрибеты по дням",
            "Распределения",
            "Сводная таблица",
            "Данные симуляций",
        ]
    )

    with tab1:
        daily_summary = (
            daily_df.groupby("day")
            .agg(
                baseline_ggr_median=("baseline_ggr", "median"),
                aor_ggr_median=("aor_ggr_gross", "median"),
                net_ggr_after_spent_median=("net_incremental_ggr_after_spent_aor_freebets", "median"),
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
            title="Медианный дневной GGR: baseline vs AOR",
            xaxis_title="День",
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
                name="Baseline ставки",
            )
        )
        fig_bets.add_trace(
            go.Scatter(
                x=daily_summary["day"],
                y=daily_summary["aor_bets_median"],
                mode="lines",
                name="AOR ставки",
            )
        )
        fig_bets.update_layout(
            title="Медианное количество ставок по дням: baseline vs AOR",
            xaxis_title="День",
            yaxis_title="Ставки",
            hovermode="x unified",
        )
        st.plotly_chart(fig_bets, use_container_width=True)

    with tab2:
        freebet_daily_summary = (
            daily_df.groupby("day")
            .agg(
                aor_issued=("aor_freebets_issued", "median"),
                aor_spent=("aor_freebets_spent", "median"),
                initial_spent=("initial_freebets_spent", "median"),
                aor_balance=("aor_freebet_balance_end", "median"),
                initial_balance=("initial_freebet_balance_end", "median"),
                total_balance=("total_freebet_balance_end", "median"),
                completed_missions=("completed_missions", "median"),
            )
            .reset_index()
        )

        fig_fb_flow = go.Figure()
        fig_fb_flow.add_trace(
            go.Scatter(
                x=freebet_daily_summary["day"],
                y=freebet_daily_summary["aor_issued"],
                mode="lines",
                name="AOR-фрибеты выданы",
            )
        )
        fig_fb_flow.add_trace(
            go.Scatter(
                x=freebet_daily_summary["day"],
                y=freebet_daily_summary["aor_spent"],
                mode="lines",
                name="AOR-фрибеты использованы",
            )
        )
        fig_fb_flow.add_trace(
            go.Scatter(
                x=freebet_daily_summary["day"],
                y=freebet_daily_summary["initial_spent"],
                mode="lines",
                name="Стартовые фрибеты использованы",
            )
        )
        fig_fb_flow.update_layout(
            title="Медианный поток фрибетов по дням",
            xaxis_title="День",
            yaxis_title="Фрибеты",
            hovermode="x unified",
        )
        st.plotly_chart(fig_fb_flow, use_container_width=True)

        fig_balance = go.Figure()
        fig_balance.add_trace(
            go.Scatter(
                x=freebet_daily_summary["day"],
                y=freebet_daily_summary["aor_balance"],
                mode="lines",
                name="AOR-баланс",
            )
        )
        fig_balance.add_trace(
            go.Scatter(
                x=freebet_daily_summary["day"],
                y=freebet_daily_summary["initial_balance"],
                mode="lines",
                name="Стартовый баланс",
            )
        )
        fig_balance.add_trace(
            go.Scatter(
                x=freebet_daily_summary["day"],
                y=freebet_daily_summary["total_balance"],
                mode="lines",
                name="Итого баланс",
            )
        )
        fig_balance.update_layout(
            title="Медианный баланс фрибетов на конец дня",
            xaxis_title="День",
            yaxis_title="Фрибеты",
            hovermode="x unified",
        )
        st.plotly_chart(fig_balance, use_container_width=True)

        fig_missions = px.bar(
            freebet_daily_summary,
            x="day",
            y="completed_missions",
            title="Медианное количество выполненных миссий по дням",
        )
        fig_missions.update_layout(xaxis_title="День", yaxis_title="Миссии")
        st.plotly_chart(fig_missions, use_container_width=True)

    with tab3:
        d1, d2 = st.columns(2)

        with d1:
            fig_roi = px.histogram(
                metrics_df,
                x="roi_after_spent",
                nbins=40,
                title="Распределение ROI по использованным AOR-фрибетам",
            )
            fig_roi.update_layout(xaxis_title="ROI", yaxis_title="Симуляции")
            st.plotly_chart(fig_roi, use_container_width=True)

        with d2:
            fig_issued = px.histogram(
                metrics_df,
                x="aor_freebets_issued_per_active_user",
                nbins=40,
                title="Распределение AOR-фрибетов, выданных на active user",
            )
            fig_issued.update_layout(xaxis_title="AOR-фрибеты выданы / active user", yaxis_title="Симуляции")
            st.plotly_chart(fig_issued, use_container_width=True)

        d3, d4 = st.columns(2)

        with d3:
            fig_spent = px.histogram(
                metrics_df,
                x="aor_freebets_spent_per_active_user",
                nbins=40,
                title="Распределение AOR-фрибетов, использованных на active user",
            )
            fig_spent.update_layout(xaxis_title="AOR-фрибеты использованы / active user", yaxis_title="Симуляции")
            st.plotly_chart(fig_spent, use_container_width=True)

        with d4:
            fig_balance = px.histogram(
                metrics_df,
                x="aor_freebets_end_balance_per_active_user",
                nbins=40,
                title="Распределение остатка AOR-фрибетов на active user",
            )
            fig_balance.update_layout(xaxis_title="AOR-фрибеты на конец / active user", yaxis_title="Симуляции")
            st.plotly_chart(fig_balance, use_container_width=True)

    with tab4:
        summary_df = build_summary(metrics_df)
        st.dataframe(format_summary(summary_df), use_container_width=True)

        st.download_button(
            "Скачать сводную таблицу CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="aor_summary_ru.csv",
            mime="text/csv",
        )

    with tab5:
        st.dataframe(metrics_df, use_container_width=True)

        st.download_button(
            "Скачать данные симуляций CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="aor_raw_simulations_ru.csv",
            mime="text/csv",
        )

    st.subheader("Логика модели")
    st.code(
        """
Базовая экономика:
average_stake = turnover / bets
hold = GGR / turnover
GGR_per_bet = GGR / bets

Распределение ставок:
user_bet_weight ~ усечённое распределение Гаусса(mean=1, std=user_bet_cv)
baseline_user_bets_30d = total_bets / active_users * user_bet_weight

Воронка AOR:
targeted_user = выбран системой
activated_user = принял миссию
completed_user = выполнил хотя бы 1 миссию

Миссии:
missions_per_completed_user = 1 + Poisson(avg_missions_per_completed_user - 1)
missions_per_completed_user <= max_missions_per_completed_user

Рост:
AOR bets = baseline bets * (1 + рост количества ставок у completed users)
AOR stake = baseline average stake * (1 + рост средней ставки у completed users)
AOR hold = baseline hold * (1 + изменение hold у completed users)

Дополнительная прибыль:
incremental_ggr_gross = AOR gross GGR - baseline GGR

Ограничение бюджета:
max_reward_budget = max(0, incremental_ggr_gross)
planned_reward_budget = max_reward_budget * reward_budget_share
planned_reward_budget <= incremental_ggr_gross

Coins и фрибеты:
1 Coin = 1 фрибет = €1 бонусного бюджета
coins_total = floor(planned_reward_budget / 1)
aor_freebets_issued_total = coins_total

Выдача по миссиям:
coins_per_completed_mission = coins_total / completed_missions_total

Стартовые фрибеты:
initial_freebets_total = initial_freebets_per_active_user * active_users

Траты фрибетов:
каждый день списывается daily_freebet_spend_rate от доступного баланса
стартовые фрибеты и AOR-фрибеты считаются отдельно

Эффективность:
program_cost_spent = использованные AOR-фрибеты * €1
program_cost_issued = выданные AOR-фрибеты * €1

net_ggr_after_spent = incremental_ggr_gross - program_cost_spent
conservative_net_ggr = incremental_ggr_gross - program_cost_issued

roi_after_spent = net_ggr_after_spent / program_cost_spent
roi_conservative = conservative_net_ggr / program_cost_issued
        """,
        language="text",
    )


if __name__ == "__main__":
    main()
