import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Tuple


DAYS = 30
COIN_COST_EUR = 1.0  # В этой версии: 1 Coin = 1 фрибет = €1 бонусного бюджета


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
    reward_budget_share: float
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
    Среднее нормализуется к 1, чтобы итоговое количество ставок совпадало
    с введённым total_bets.
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

    user_bets_30d = (inputs.total_bets / inputs.active_users) * user_weights
    baseline_user_day_bets = np.outer(user_bets_30d, daily_weights)

    scale_to_total = safe_divide(inputs.total_bets, baseline_user_day_bets.sum(), default=1.0)
    baseline_user_day_bets = baseline_user_day_bets * scale_to_total

    targeted = rng.random(inputs.active_users) < inputs.target_share
    activated = targeted & (rng.random(inputs.active_users) < inputs.activation_rate)
    completed = activated & (rng.random(inputs.active_users) < inputs.completion_rate)

    bet_multiplier = np.where(completed, 1.0 + inputs.bet_count_uplift, 1.0)
    stake_multiplier = np.where(completed, 1.0 + inputs.avg_stake_uplift, 1.0)
    hold_multiplier = np.where(completed, 1.0 + inputs.hold_uplift, 1.0)

    aor_user_day_bets = baseline_user_day_bets * bet_multiplier[:, None]

    baseline_user_day_turnover = baseline_user_day_bets * base_avg_stake
    aor_user_day_turnover = aor_user_day_bets * (base_avg_stake * stake_multiplier[:, None])

    baseline_user_day_ggr = baseline_user_day_turnover * base_hold
    aor_user_day_ggr = aor_user_day_turnover * (base_hold * hold_multiplier[:, None])

    baseline_bets = baseline_user_day_bets.sum()
    aor_bets = aor_user_day_bets.sum()
    baseline_turnover = baseline_user_day_turnover.sum()
    aor_turnover = aor_user_day_turnover.sum()
    baseline_ggr = baseline_user_day_ggr.sum()
    aor_ggr_gross = aor_user_day_ggr.sum()

    incremental_ggr_gross = aor_ggr_gross - baseline_ggr

    # Главное ограничение модели:
    # фонд Coins / фрибетов не может превышать дополнительную прибыль от программы.
    max_reward_budget = max(0.0, incremental_ggr_gross)
    planned_reward_budget = max_reward_budget * inputs.reward_budget_share

    max_coins_total = np.floor(max_reward_budget / COIN_COST_EUR)
    coins_total = np.floor(planned_reward_budget / COIN_COST_EUR)

    # Курс: 1 Coin = 1 фрибет.
    max_freebets_total = max_coins_total
    freebets_total = coins_total

    program_cost = coins_total * COIN_COST_EUR
    incremental_ggr_net = incremental_ggr_gross - program_cost

    roi = safe_divide(incremental_ggr_net, program_cost, default=np.nan)
    gross_uplift_pct = safe_divide(incremental_ggr_gross, baseline_ggr, default=np.nan)
    net_uplift_pct = safe_divide(incremental_ggr_net, baseline_ggr, default=np.nan)

    targeted_users = int(targeted.sum())
    activated_users = int(activated.sum())
    completed_users = int(completed.sum())

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

        "max_reward_budget": max_reward_budget,
        "planned_reward_budget": planned_reward_budget,
        "program_cost": program_cost,

        "max_coins_total": max_coins_total,
        "coins_total": coins_total,
        "max_freebets_total": max_freebets_total,
        "freebets_total": freebets_total,

        "coins_per_active_user": safe_divide(coins_total, inputs.active_users),
        "coins_per_targeted_user": safe_divide(coins_total, targeted_users),
        "coins_per_completed_user": safe_divide(coins_total, completed_users),

        "max_coins_per_active_user": safe_divide(max_coins_total, inputs.active_users),
        "max_coins_per_targeted_user": safe_divide(max_coins_total, targeted_users),
        "max_coins_per_completed_user": safe_divide(max_coins_total, completed_users),

        "freebets_per_active_user": safe_divide(freebets_total, inputs.active_users),
        "max_freebets_per_active_user": safe_divide(max_freebets_total, inputs.active_users),

        "incremental_ggr_net": incremental_ggr_net,
        "roi": roi,
        "gross_uplift_pct": gross_uplift_pct,
        "net_uplift_pct": net_uplift_pct,

        "reward_budget_does_not_exceed_incremental_profit": program_cost <= max(0.0, incremental_ggr_gross),
        "break_even_after_rewards": incremental_ggr_net >= 0,
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

    return metrics, daily


@st.cache_data(show_spinner=False)
def run_simulations(inputs_dict: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inputs = ModelInputs(**inputs_dict)
    rng = np.random.default_rng(inputs.random_seed)

    metrics = []
    daily_frames = []

    for sim_id in range(1, inputs.n_simulations + 1):
        row, daily = run_single_simulation(rng, inputs, sim_id)
        metrics.append(row)
        daily_frames.append(daily)

    return pd.DataFrame(metrics), pd.concat(daily_frames, ignore_index=True)


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_order = [
        "completed_users",
        "incremental_bets",
        "incremental_turnover",
        "incremental_ggr_gross",
        "max_reward_budget",
        "planned_reward_budget",
        "program_cost",
        "incremental_ggr_net",
        "roi",
        "gross_uplift_pct",
        "net_uplift_pct",
        "max_coins_total",
        "coins_total",
        "max_coins_per_active_user",
        "coins_per_active_user",
        "freebets_per_active_user",
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
        "completed_users": "Пользователи, выполнившие миссию",
        "incremental_bets": "Дополнительные ставки",
        "incremental_turnover": "Дополнительный оборот",
        "incremental_ggr_gross": "Дополнительный gross GGR",
        "max_reward_budget": "Максимальный фонд Coins / фрибетов",
        "planned_reward_budget": "Плановый фонд Coins / фрибетов",
        "program_cost": "Стоимость программы",
        "incremental_ggr_net": "Net incremental GGR",
        "roi": "ROI программы",
        "gross_uplift_pct": "Gross uplift к baseline GGR",
        "net_uplift_pct": "Net uplift к baseline GGR",
        "max_coins_total": "Максимум Coins всего",
        "coins_total": "Расчётные Coins всего",
        "max_coins_per_active_user": "Максимум Coins на active user",
        "coins_per_active_user": "Расчётные Coins на active user",
        "freebets_per_active_user": "Фрибеты на active user",
    }

    money_metrics = {
        "incremental_turnover",
        "incremental_ggr_gross",
        "max_reward_budget",
        "planned_reward_budget",
        "program_cost",
        "incremental_ggr_net",
    }
    pct_metrics = {
        "roi",
        "gross_uplift_pct",
        "net_uplift_pct",
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
        page_title="AOR — расчёт Coins и фрибетов",
        page_icon="📈",
        layout="wide",
    )

    st.title("Art of Retention — симуляционная модель с расчётом Coins и фрибетов")
    st.caption(
        "Coins и фрибеты больше не вводятся вручную. Модель рассчитывает их автоматически "
        "из дополнительного GGR, созданного программой. Фонд вознаграждений не может превышать "
        "дополнительную прибыль от применения AOR."
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
        completion_rate = st.slider("Доля пользователей, выполнивших миссию", 0.0, 1.0, 0.60, 0.01)

        st.header("3. Характеристики роста")
        bet_count_uplift = st.slider("Рост количества ставок у выполнивших миссию", -0.50, 2.00, 0.15, 0.01)
        avg_stake_uplift = st.slider("Рост средней ставки у выполнивших миссию", -0.50, 2.00, 0.05, 0.01)
        hold_uplift = st.slider("Изменение hold / GGR margin у выполнивших миссию", -0.50, 1.00, 0.00, 0.01)

        st.header("4. Правило расчёта Coins")
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

        st.header("5. Настройки симуляции")
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Средняя ставка", eur(base_avg_stake))
    c2.metric("GGR margin / hold", pct(base_hold))
    c3.metric("GGR на одну ставку", eur(base_ggr_per_bet))
    c4.metric("Ставок на active user", number(bets_per_user))

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
        reward_budget_share=float(reward_budget_share),
        user_bet_cv=float(user_bet_cv),
        daily_volatility=float(daily_volatility),
        random_seed=int(random_seed),
    )

    metrics_df, daily_df = run_simulations(inputs.__dict__)

    st.subheader("Результаты симуляции")
    avg_freebets = metrics_df["freebets_per_active_user"].mean()
    avg_max_coins = metrics_df["max_coins_per_active_user"].mean()
    median_roi = metrics_df["roi"].median()
    mean_net = metrics_df["incremental_ggr_net"].mean()
    positive_share = metrics_df["break_even_after_rewards"].mean()
    constraint_share = metrics_df["reward_budget_does_not_exceed_incremental_profit"].mean()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Фрибеты / active user", number(avg_freebets))
    m2.metric("Макс. Coins / active user", number(avg_max_coins))
    m3.metric("Медианный ROI", pct(median_roi))
    m4.metric("Средний net GGR", eur(mean_net))
    m5.metric("Прибыльные симуляции", pct(positive_share))
    m6.metric("Лимит соблюдён", pct(constraint_share))

    st.markdown(
        """
        **Как теперь считаются Coins и фрибеты.**

        Сначала модель считает дополнительный gross GGR от AOR. После этого определяется максимальный фонд вознаграждений:

        ```text
        Max Reward Budget = max(0, Incremental Gross GGR)
        ```

        Плановый фонд Coins считается как доля от дополнительной прибыли:

        ```text
        Planned Reward Budget = Max Reward Budget × Reward Budget Share
        ```

        В этой версии используется фиксированное правило:

        ```text
        1 Coin = 1 фрибет = €1 бонусного бюджета
        ```

        Поэтому:

        ```text
        Coins = floor(Planned Reward Budget / €1)
        Freebets = Coins
        ```

        За счёт этого стоимость Coins / фрибетов никогда не превышает дополнительную прибыль от программы.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Динамика за 30 дней",
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
        d1, d2 = st.columns(2)

        with d1:
            fig_roi = px.histogram(
                metrics_df,
                x="roi",
                nbins=40,
                title="Распределение ROI программы",
            )
            fig_roi.update_layout(xaxis_title="ROI", yaxis_title="Симуляции")
            st.plotly_chart(fig_roi, use_container_width=True)

        with d2:
            fig_freebets = px.histogram(
                metrics_df,
                x="freebets_per_active_user",
                nbins=40,
                title="Распределение расчётных фрибетов на active user",
            )
            fig_freebets.update_layout(
                xaxis_title="Фрибеты на active user",
                yaxis_title="Симуляции",
            )
            st.plotly_chart(fig_freebets, use_container_width=True)

        d3, d4 = st.columns(2)

        with d3:
            fig_max_coins = px.histogram(
                metrics_df,
                x="max_coins_per_active_user",
                nbins=40,
                title="Распределение максимальных Coins на active user",
            )
            fig_max_coins.update_layout(
                xaxis_title="Максимум Coins на active user",
                yaxis_title="Симуляции",
            )
            st.plotly_chart(fig_max_coins, use_container_width=True)

        with d4:
            fig_net = px.histogram(
                metrics_df,
                x="incremental_ggr_net",
                nbins=40,
                title="Распределение net incremental GGR",
            )
            fig_net.update_layout(xaxis_title="Net incremental GGR, €", yaxis_title="Симуляции")
            st.plotly_chart(fig_net, use_container_width=True)

    with tab3:
        summary_df = build_summary(metrics_df)
        st.dataframe(format_summary(summary_df), use_container_width=True)

        st.download_button(
            "Скачать сводную таблицу CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="aor_summary_ru.csv",
            mime="text/csv",
        )

    with tab4:
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

Симуляция на уровне пользователей:
user_bet_weight ~ усечённое распределение Гаусса(mean=1, std=user_bet_cv)
baseline_user_bets_30d = total_bets / active_users * user_bet_weight

Воронка AOR:
completed_user = targeted * activated * completed

Рост:
AOR bets = baseline bets * (1 + рост количества ставок у выполнивших миссию)
AOR stake = baseline average stake * (1 + рост средней ставки у выполнивших миссию)
AOR hold = baseline hold * (1 + изменение hold у выполнивших миссию)

Дополнительная прибыль:
incremental_ggr_gross = AOR gross GGR - baseline GGR

Ограничение бюджета:
max_reward_budget = max(0, incremental_ggr_gross)
planned_reward_budget = max_reward_budget * reward_budget_share
planned_reward_budget <= incremental_ggr_gross

Coins и фрибеты:
1 Coin = 1 фрибет = €1 бонусного бюджета
coins_total = floor(planned_reward_budget / 1)
freebets_total = coins_total

Эффективность:
program_cost = coins_total
incremental_ggr_net = incremental_ggr_gross - program_cost
ROI = incremental_ggr_net / program_cost
        """,
        language="text",
    )


if __name__ == "__main__":
    main()
