import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="AOR freebet simulation",
    page_icon="🎲",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------

def format_eur(value: float) -> str:
    return f"€{value:,.0f}".replace(",", " ")


def format_num(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


@st.cache_data(show_spinner=False)
def run_simulation(
    bets_count: int,
    turnover: float,
    ggr: float,
    active_users: int,
    simulations: int,
    allocation_pool_pct: float,
    ggr_cv: float,
    util_mean: float,
    util_std: float,
    avg_freebet_odds: float,
    freebet_margin: float,
    payout_std: float,
    withdrawal_rate: float,
    random_seed: int,
    sample_per_sim: int,
):
    """Run Monte Carlo simulation for AOR freebet allocation and spend.

    Model logic:
    1. Build synthetic user GGR weights using Gaussian distribution.
    2. Allocate total freebet pool proportionally to those weights.
    3. Simulate freebet utilization with Gaussian distribution.
    4. Estimate user winnings from freebet bets and assume users withdraw them.
    """

    rng = np.random.default_rng(random_seed)

    pool = max(ggr, 0) * allocation_pool_pct
    avg_ggr_per_user = safe_div(ggr, active_users)
    ggr_std = abs(avg_ggr_per_user) * ggr_cv

    # Expected user cash payout from a freebet when stake is not returned.
    # For decimal odds O and sportsbook margin m:
    # expected cash payout per 1 EUR freebet = (1 - m) * (O - 1) / O
    expected_payout_multiplier = max(0.0, (1.0 - freebet_margin) * (avg_freebet_odds - 1.0) / avg_freebet_odds)

    summary_rows = []
    sampled_rows = []

    # Minimum positive weight so every active user receives a non-zero proportional allocation.
    eps_weight = max(avg_ggr_per_user * 0.001, 0.000001)

    for sim_id in range(1, simulations + 1):
        # Gaussian user-level GGR proxy. It is clipped to a small positive value because
        # proportional freebet allocation cannot use negative weights.
        ggr_weight = rng.normal(loc=avg_ggr_per_user, scale=ggr_std, size=active_users)
        ggr_weight = np.clip(ggr_weight, eps_weight, None)

        freebet_allocated = pool * ggr_weight / ggr_weight.sum()

        # Gaussian utilization: share of granted freebets used within 30 days.
        utilization = rng.normal(loc=util_mean, scale=util_std, size=active_users)
        utilization = np.clip(utilization, 0.0, 1.0)

        freebet_spent = freebet_allocated * utilization

        # Gaussian payout intensity for winnings from freebet bets.
        # This represents volatility around expected freebet winnings.
        payout_multiplier = rng.normal(
            loc=expected_payout_multiplier,
            scale=payout_std,
            size=active_users,
        )
        payout_multiplier = np.clip(payout_multiplier, 0.0, None)

        user_winnings = freebet_spent * payout_multiplier
        withdrawn_winnings = user_winnings * withdrawal_rate

        allocated_total = float(freebet_allocated.sum())
        spent_total = float(freebet_spent.sum())
        unused_total = allocated_total - spent_total
        winnings_total = float(user_winnings.sum())
        withdrawn_total = float(withdrawn_winnings.sum())
        operator_net_cost = spent_total + withdrawn_total
        remaining_ggr_after_pool_and_wins = ggr - operator_net_cost

        summary_rows.append(
            {
                "simulation": sim_id,
                "allocated_freebets": allocated_total,
                "spent_freebets": spent_total,
                "unused_freebets": unused_total,
                "utilization_rate": safe_div(spent_total, allocated_total),
                "user_winnings_from_freebets": winnings_total,
                "withdrawn_winnings": withdrawn_total,
                "operator_net_cost": operator_net_cost,
                "remaining_ggr_after_freebets": remaining_ggr_after_pool_and_wins,
                "avg_freebet_per_user": float(np.mean(freebet_allocated)),
                "median_freebet_per_user": float(np.median(freebet_allocated)),
                "p90_freebet_per_user": float(np.percentile(freebet_allocated, 90)),
                "p95_freebet_per_user": float(np.percentile(freebet_allocated, 95)),
            }
        )

        # Keep a sample for distribution charts to avoid creating very large dataframes
        # when active_users * simulations is huge.
        sample_size = min(sample_per_sim, active_users)
        sample_idx = rng.choice(active_users, size=sample_size, replace=False)
        sampled_rows.append(
            pd.DataFrame(
                {
                    "simulation": sim_id,
                    "freebet_allocated": freebet_allocated[sample_idx],
                    "freebet_spent": freebet_spent[sample_idx],
                    "user_winnings_from_freebets": user_winnings[sample_idx],
                    "utilization_rate": utilization[sample_idx],
                }
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    sample_df = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else pd.DataFrame()

    return summary_df, sample_df, expected_payout_multiplier


# -----------------------------
# Sidebar inputs
# -----------------------------

st.title("AOR: модель распределения и использования фрибетов")
st.caption(
    "Monte Carlo-модель на 30 дней: начисление фрибетов из пула 20% GGR, "
    "пропорциональное распределение по синтетическому user-level GGR и симуляция траты фрибетов."
)

with st.sidebar:
    st.header("Входные данные за 30 дней")

    bets_count = st.number_input(
        "Количество ставок за период",
        min_value=1,
        value=5_483_458,
        step=10_000,
        format="%d",
    )
    turnover = st.number_input(
        "Оборот, € за период",
        min_value=1.0,
        value=40_834_160.0,
        step=100_000.0,
        format="%.2f",
    )
    ggr = st.number_input(
        "GGR, € за период",
        min_value=0.0,
        value=4_682_582.0,
        step=50_000.0,
        format="%.2f",
    )
    active_users = st.number_input(
        "Активных пользователей за период",
        min_value=1,
        value=372_326,
        step=1_000,
        format="%d",
    )

    st.divider()
    st.header("Параметры симуляции")

    simulations = st.slider(
        "Количество симуляций",
        min_value=1,
        max_value=300,
        value=300,
        step=1,
    )
    allocation_pool_pct = st.slider(
        "Пул фрибетов от GGR прошлого периода",
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.01,
        format="%.2f",
        help="0.20 означает, что на начисление фрибетов идет 20% GGR предыдущих 30 дней.",
    )
    ggr_cv = st.slider(
        "Разброс user-level GGR, CV",
        min_value=0.05,
        max_value=3.0,
        value=1.0,
        step=0.05,
        help="CV = стандартное отклонение / среднее. Чем выше значение, тем сильнее различаются пользователи по GGR.",
    )

    util_mean = st.slider(
        "Средняя доля использованных фрибетов",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.01,
    )
    util_std = st.slider(
        "Разброс доли использования фрибетов",
        min_value=0.0,
        max_value=0.5,
        value=0.20,
        step=0.01,
    )

    avg_freebet_odds = st.slider(
        "Средний коэффициент ставки на фрибет",
        min_value=1.01,
        max_value=10.0,
        value=2.00,
        step=0.01,
    )
    freebet_margin = st.slider(
        "Маржа на ставках с фрибетом",
        min_value=0.0,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Используется для оценки ожидаемого выигрыша пользователя с фрибета.",
    )
    payout_std = st.slider(
        "Волатильность выигрыша с фрибетов",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
    )
    withdrawal_rate = st.slider(
        "Доля выигрыша, которую пользователь забирает",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
    )

    st.divider()
    st.header("Технические параметры")

    sample_per_sim = st.slider(
        "Сэмпл пользователей на симуляцию для графиков",
        min_value=100,
        max_value=10_000,
        value=1_000,
        step=100,
        help="Нужен, чтобы диаграммы строились быстро даже при сотнях тысяч пользователей.",
    )
    random_seed = st.number_input(
        "Random seed",
        min_value=1,
        value=42,
        step=1,
        format="%d",
    )


# -----------------------------
# Derived input metrics
# -----------------------------

avg_stake = safe_div(turnover, bets_count)
ggr_margin = safe_div(ggr, turnover)
active_user_turnover = safe_div(turnover, active_users)
active_user_ggr = safe_div(ggr, active_users)
freebet_pool = ggr * allocation_pool_pct
avg_pool_per_user = safe_div(freebet_pool, active_users)

input_cols = st.columns(5)
input_cols[0].metric("Средний размер ставки", format_eur(avg_stake))
input_cols[1].metric("GGR margin", f"{ggr_margin:.2%}")
input_cols[2].metric("Оборот / user", format_eur(active_user_turnover))
input_cols[3].metric("GGR / user", format_eur(active_user_ggr))
input_cols[4].metric("Пул фрибетов", format_eur(freebet_pool))

with st.expander("Математика модели", expanded=False):
    st.markdown(
        """
**1. Пул фрибетов**

`Freebet Pool = GGR_previous_30d × Pool %`

По умолчанию `Pool % = 20%`.

**2. Синтетический GGR пользователя**

Так как на входе есть только агрегированные данные, user-level GGR создаётся через нормальное распределение:

`User GGR Weightᵢ ~ Normal(AVG_GGR_per_user, AVG_GGR_per_user × CV)`

Отрицательные значения обрезаются до малого положительного значения, чтобы каждый активный пользователь получил ненулевой вес для распределения.

**3. Начисление фрибета пользователю**

`Freebetᵢ = Freebet Pool × User GGR Weightᵢ / Sum(User GGR Weight)`

Так весь пул фрибетов строго равен заданной доле GGR прошлого периода.

**4. Использование фрибетов**

`Utilizationᵢ ~ Normal(Mean Utilization, Utilization Std)`, затем значение обрезается в диапазон от 0 до 1.

`Spent Freebetᵢ = Freebetᵢ × Utilizationᵢ`

**5. Выигрыш пользователя с фрибета**

По умолчанию считается, что freebet stake не возвращается, а пользователь забирает выигрыш:

`Expected Payout Multiplier = (1 - Margin) × (Odds - 1) / Odds`

`User Winningsᵢ = Spent Freebetᵢ × Gaussian Payout Multiplierᵢ`

`Withdrawn Winningsᵢ = User Winningsᵢ × Withdrawal Rate`
        """
    )


# -----------------------------
# Run simulation
# -----------------------------

if active_users * simulations > 120_000_000:
    st.warning(
        "Очень большая симуляция: active users × simulations превышает 120 млн. "
        "Расчёт может занять заметное время. Уменьшите количество симуляций или пользователей, если дашборд работает медленно."
    )

with st.spinner("Считаю симуляции..."):
    summary_df, sample_df, expected_payout_multiplier = run_simulation(
        bets_count=int(bets_count),
        turnover=float(turnover),
        ggr=float(ggr),
        active_users=int(active_users),
        simulations=int(simulations),
        allocation_pool_pct=float(allocation_pool_pct),
        ggr_cv=float(ggr_cv),
        util_mean=float(util_mean),
        util_std=float(util_std),
        avg_freebet_odds=float(avg_freebet_odds),
        freebet_margin=float(freebet_margin),
        payout_std=float(payout_std),
        withdrawal_rate=float(withdrawal_rate),
        random_seed=int(random_seed),
        sample_per_sim=int(sample_per_sim),
    )


# -----------------------------
# Results
# -----------------------------

st.subheader("Результаты симуляции")

median_row = summary_df.median(numeric_only=True)

result_cols = st.columns(5)
result_cols[0].metric("Начислено фрибетов", format_eur(median_row["allocated_freebets"]))
result_cols[1].metric("Потрачено фрибетов", format_eur(median_row["spent_freebets"]))
result_cols[2].metric("Utilization", f"{median_row['utilization_rate']:.1%}")
result_cols[3].metric("Выигрыш user с фрибетов", format_eur(median_row["user_winnings_from_freebets"]))
result_cols[4].metric("Net cost для оператора", format_eur(median_row["operator_net_cost"]))

result_cols_2 = st.columns(5)
result_cols_2[0].metric("Средний фрибет / user", format_eur(median_row["avg_freebet_per_user"]))
result_cols_2[1].metric("Медианный фрибет / user", format_eur(median_row["median_freebet_per_user"]))
result_cols_2[2].metric("P90 фрибет / user", format_eur(median_row["p90_freebet_per_user"]))
result_cols_2[3].metric("P95 фрибет / user", format_eur(median_row["p95_freebet_per_user"]))
result_cols_2[4].metric("GGR после freebet cost", format_eur(median_row["remaining_ggr_after_freebets"]))

st.info(
    f"Ожидаемый cash payout multiplier по фрибету: {expected_payout_multiplier:.3f}. "
    f"Это означает, что при текущих параметрах каждый €1 потраченного фрибета в среднем создаёт "
    f"около €{expected_payout_multiplier:.2f} выигрыша пользователя до применения withdrawal rate."
)

# Charts
chart_tab_1, chart_tab_2, chart_tab_3, chart_tab_4 = st.tabs(
    [
        "Распределение фрибетов",
        "Использование фрибетов",
        "Итоги по симуляциям",
        "Данные",
    ]
)

with chart_tab_1:
    st.markdown("#### Распределение начисленных фрибетов по пользователям")
    fig_alloc = px.histogram(
        sample_df,
        x="freebet_allocated",
        nbins=80,
        marginal="box",
        labels={"freebet_allocated": "Начисленный фрибет, €"},
        title="Распределение начисленных фрибетов, user-level sample",
    )
    fig_alloc.update_layout(bargap=0.02)
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.markdown("#### Распределение фрибетов по percentiles")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_df = pd.DataFrame(
        {
            "percentile": percentiles,
            "freebet_allocated": [np.percentile(sample_df["freebet_allocated"], p) for p in percentiles],
            "freebet_spent": [np.percentile(sample_df["freebet_spent"], p) for p in percentiles],
            "user_winnings_from_freebets": [np.percentile(sample_df["user_winnings_from_freebets"], p) for p in percentiles],
        }
    )
    st.dataframe(
        pct_df.style.format(
            {
                "freebet_allocated": "€{:,.2f}",
                "freebet_spent": "€{:,.2f}",
                "user_winnings_from_freebets": "€{:,.2f}",
            }
        ),
        use_container_width=True,
    )

with chart_tab_2:
    st.markdown("#### Распределение потраченных фрибетов")
    fig_spent = px.histogram(
        sample_df,
        x="freebet_spent",
        nbins=80,
        marginal="box",
        labels={"freebet_spent": "Потраченный фрибет, €"},
        title="Распределение потраченных фрибетов, user-level sample",
    )
    fig_spent.update_layout(bargap=0.02)
    st.plotly_chart(fig_spent, use_container_width=True)

    st.markdown("#### Распределение выигрышей пользователей с фрибетов")
    fig_win = px.histogram(
        sample_df,
        x="user_winnings_from_freebets",
        nbins=80,
        marginal="box",
        labels={"user_winnings_from_freebets": "Выигрыш пользователя, €"},
        title="Распределение выигрышей с фрибетов, user-level sample",
    )
    fig_win.update_layout(bargap=0.02)
    st.plotly_chart(fig_win, use_container_width=True)

with chart_tab_3:
    st.markdown("#### Распределение итоговой стоимости программы по симуляциям")
    fig_cost = px.histogram(
        summary_df,
        x="operator_net_cost",
        nbins=min(60, max(10, simulations)),
        marginal="box",
        labels={"operator_net_cost": "Net cost для оператора, €"},
        title="Net cost = потраченные фрибеты + забранный выигрыш",
    )
    fig_cost.update_layout(bargap=0.02)
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("#### Динамика результатов по симуляциям")
    line_df = summary_df[
        [
            "simulation",
            "spent_freebets",
            "withdrawn_winnings",
            "operator_net_cost",
            "remaining_ggr_after_freebets",
        ]
    ].melt(id_vars="simulation", var_name="metric", value_name="value")

    fig_line = px.line(
        line_df,
        x="simulation",
        y="value",
        color="metric",
        labels={"simulation": "Симуляция", "value": "€", "metric": "Метрика"},
        title="Итоги по каждой симуляции",
    )
    st.plotly_chart(fig_line, use_container_width=True)

with chart_tab_4:
    st.markdown("#### Summary по 300 симуляциям")
    st.dataframe(
        summary_df.style.format(
            {
                "allocated_freebets": "€{:,.0f}",
                "spent_freebets": "€{:,.0f}",
                "unused_freebets": "€{:,.0f}",
                "utilization_rate": "{:.1%}",
                "user_winnings_from_freebets": "€{:,.0f}",
                "withdrawn_winnings": "€{:,.0f}",
                "operator_net_cost": "€{:,.0f}",
                "remaining_ggr_after_freebets": "€{:,.0f}",
                "avg_freebet_per_user": "€{:,.2f}",
                "median_freebet_per_user": "€{:,.2f}",
                "p90_freebet_per_user": "€{:,.2f}",
                "p95_freebet_per_user": "€{:,.2f}",
            }
        ),
        use_container_width=True,
        height=420,
    )

    csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать summary CSV",
        data=csv,
        file_name="aor_freebet_simulation_summary.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Важно: это агрегированная модель. Так как нет реального user-level GGR, пользовательское распределение создаётся синтетически через Gaussian weights. "
    "Для production-модели лучше заменить synthetic GGR на реальные user-level данные: GGR, turnover, bet count, сегмент, VIP/churn score и историю бонусов."
)
