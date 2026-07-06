import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AOR bookmaker-realistic simulation",
    page_icon="🎲",
    layout="wide",
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator is None or denominator == 0:
        return default
    return numerator / denominator


def format_eur(value: float, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"€{value:,.{digits}f}".replace(",", " ")


def format_num(value: float, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ")


def format_pct(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}%}"


def payout_multiplier_from_freebet(avg_odds: float, margin: float, stake_returned: bool) -> float:
    """Expected cash payout per €1 of spent freebet.

    stake_returned=False is the common "stake not returned" freebet mechanic:
    the freebet stake is a bonus token, and only winnings can become cash.

    stake_returned=True is more expensive and behaves closer to a normal cash stake.
    """
    avg_odds = max(avg_odds, 1.01)
    margin = np.clip(margin, 0.0, 0.95)

    if stake_returned:
        return max(0.0, 1.0 - margin)

    return max(0.0, (1.0 - margin) * (avg_odds - 1.0) / avg_odds)


def build_user_ggr_distribution(
    rng: np.random.Generator,
    active_users: int,
    total_ggr: float,
    mode: str,
    gaussian_cv: float,
    lognormal_sigma: float,
    negative_user_share: float,
    negative_loss_pool_pct: float,
) -> np.ndarray:
    """Create synthetic user-level GGR that sums to total GGR.

    The dashboard can run in two modes:
    - Gaussian: closer to the original task, keeps negative users if the distribution creates them.
    - Skewed bookmaker tail: more realistic for sportsbooks, where a small share of users produces a large part of GGR.
    """

    active_users = int(max(active_users, 1))
    total_ggr = float(total_ggr)

    if total_ggr <= 0:
        return np.zeros(active_users, dtype=float)

    avg_ggr = total_ggr / active_users

    if mode.startswith("Gaussian"):
        std = abs(avg_ggr) * gaussian_cv
        raw = rng.normal(loc=avg_ggr, scale=std, size=active_users)

        # Re-center the distribution so the aggregate matches the input GGR.
        raw = raw - raw.mean() + avg_ggr

        # Numerical guard: after re-centering, the sum should equal total_ggr.
        if abs(raw.sum()) > 1e-9:
            raw = raw * (total_ggr / raw.sum())

        return raw

    # Skewed bookmaker-like profile:
    # positive long-tail GGR users minus a controlled pool of negative-GGR users.
    n_neg = int(np.clip(round(active_users * negative_user_share), 0, active_users - 1))
    n_pos = active_users - n_neg

    positive_pool = total_ggr * (1.0 + max(0.0, negative_loss_pool_pct))
    negative_pool = total_ggr * max(0.0, negative_loss_pool_pct)

    positive_weights = rng.lognormal(mean=0.0, sigma=max(0.05, lognormal_sigma), size=n_pos)
    positive_values = positive_pool * positive_weights / positive_weights.sum()

    values = np.empty(active_users, dtype=float)
    values[:n_pos] = positive_values

    if n_neg > 0 and negative_pool > 0:
        negative_weights = rng.gamma(shape=1.5, scale=1.0, size=n_neg)
        negative_values = -negative_pool * negative_weights / negative_weights.sum()
        values[n_pos:] = negative_values

    rng.shuffle(values)

    # Final guard to keep aggregate exactly aligned with the input.
    values = values + (total_ggr - values.sum()) / active_users
    return values


def reallocate_remaining_budget(
    grant: np.ndarray,
    max_allowed: np.ndarray,
    weights: np.ndarray,
    budget_ceiling: float,
    min_freebet: float,
    max_passes: int = 8,
) -> np.ndarray:
    """Optional helper to spend more of the budget without breaking caps or ROI constraints."""
    grant = grant.copy()

    for _ in range(max_passes):
        remaining = budget_ceiling - grant.sum()
        if remaining <= 1e-9:
            break

        capacity = np.maximum(0.0, max_allowed - grant)
        eligible = capacity > 1e-9

        if not np.any(eligible):
            break

        allocation_weights = np.where(eligible, weights, 0.0)
        if allocation_weights.sum() <= 0:
            allocation_weights = eligible.astype(float)

        increment = remaining * allocation_weights / allocation_weights.sum()
        increment = np.minimum(increment, capacity)
        grant += increment

    if min_freebet > 0:
        grant = np.where(grant >= min_freebet, grant, 0.0)

    return grant


@st.cache_data(show_spinner=False)
def run_simulation(
    bets_count: int,
    turnover: float,
    ggr: float,
    active_users: int,
    simulations: int,
    ggr_distribution_mode: str,
    allocation_pool_pct: float,
    gaussian_ggr_cv: float,
    lognormal_sigma: float,
    negative_user_share: float,
    negative_loss_pool_pct: float,
    allocation_noise_std: float,
    utilization_mean: float,
    utilization_std: float,
    avg_freebet_odds: float,
    freebet_margin: float,
    stake_returned: bool,
    payout_std: float,
    withdrawal_rate: float,
    recycled_winnings_turnover_multiplier: float,
    expected_uplift_mean: float,
    expected_uplift_std: float,
    response_mean: float,
    response_std: float,
    max_cash_cost_to_incremental_ggr: float,
    min_expected_profit_per_user: float,
    min_freebet_per_user: float,
    max_freebet_per_user: float,
    restricted_share: float,
    bonus_abuse_share: float,
    force_spend_budget: bool,
    random_seed: int,
    sample_per_sim: int,
):
    rng = np.random.default_rng(random_seed)

    simulations = int(np.clip(simulations, 1, 300))
    active_users = int(max(active_users, 1))
    ggr = float(max(ggr, 0.0))
    turnover = float(max(turnover, 0.0))

    ggr_margin = safe_div(ggr, turnover, 0.0)
    avg_ggr_per_user = safe_div(ggr, active_users, 0.0)

    budget_ceiling = ggr * allocation_pool_pct
    expected_payout_multiplier = payout_multiplier_from_freebet(
        avg_odds=avg_freebet_odds,
        margin=freebet_margin,
        stake_returned=stake_returned,
    )

    expected_cash_cost_per_granted_eur = (
        utilization_mean * expected_payout_multiplier * withdrawal_rate
    )

    summary_rows = []
    sampled_rows = []

    for sim_id in range(1, simulations + 1):
        user_ggr_prev = build_user_ggr_distribution(
            rng=rng,
            active_users=active_users,
            total_ggr=ggr,
            mode=ggr_distribution_mode,
            gaussian_cv=gaussian_ggr_cv,
            lognormal_sigma=lognormal_sigma,
            negative_user_share=negative_user_share,
            negative_loss_pool_pct=negative_loss_pool_pct,
        )

        positive_ggr = np.clip(user_ggr_prev, 0.0, None)

        restricted = rng.random(active_users) < restricted_share
        bonus_abuser = rng.random(active_users) < bonus_abuse_share

        # Candidate pool: the model excludes restricted, bonus-abuse and negative-GGR users.
        candidate = (positive_ggr > 0) & (~restricted) & (~bonus_abuser)

        allocation_noise = rng.normal(loc=1.0, scale=allocation_noise_std, size=active_users)
        allocation_noise = np.clip(allocation_noise, 0.05, 5.0)

        weights = np.where(candidate, positive_ggr * allocation_noise, 0.0)

        grant = np.zeros(active_users, dtype=float)

        # User-specific expected uplift and expected response are used for NO_ACTION decisions.
        expected_uplift = rng.normal(
            loc=expected_uplift_mean,
            scale=expected_uplift_std,
            size=active_users,
        )
        expected_uplift = np.clip(expected_uplift, 0.0, 5.0)

        expected_response = rng.normal(loc=response_mean, scale=response_std, size=active_users)
        expected_response = np.clip(expected_response, 0.0, 1.5)

        expected_incremental_ggr = positive_ggr * expected_uplift * expected_response

        if budget_ceiling > 0 and weights.sum() > 0:
            raw_grant = budget_ceiling * weights / weights.sum()

            if expected_cash_cost_per_granted_eur > 0:
                max_grant_by_roi = (
                    expected_incremental_ggr
                    * max_cash_cost_to_incremental_ggr
                    / expected_cash_cost_per_granted_eur
                )
            else:
                max_grant_by_roi = np.full(active_users, max_freebet_per_user)

            max_allowed = np.minimum(max_freebet_per_user, max_grant_by_roi)
            max_allowed = np.where(candidate, np.maximum(max_allowed, 0.0), 0.0)

            grant = np.minimum(raw_grant, max_allowed)

            expected_cash_cost = grant * expected_cash_cost_per_granted_eur
            expected_profit = expected_incremental_ggr - expected_cash_cost

            grant = np.where(expected_profit >= min_expected_profit_per_user, grant, 0.0)
            grant = np.where(grant >= min_freebet_per_user, grant, 0.0)

            if force_spend_budget:
                # Only reallocate to users that still pass the ROI cap.
                grant = reallocate_remaining_budget(
                    grant=grant,
                    max_allowed=max_allowed,
                    weights=weights,
                    budget_ceiling=budget_ceiling,
                    min_freebet=min_freebet_per_user,
                )

        offered = grant > 0

        utilization = rng.normal(
            loc=utilization_mean,
            scale=utilization_std,
            size=active_users,
        )
        utilization = np.clip(utilization, 0.0, 1.0)
        utilization = np.where(offered, utilization, 0.0)

        spent_face_value = grant * utilization
        burned_face_value = grant - spent_face_value

        payout_multiplier = rng.normal(
            loc=expected_payout_multiplier,
            scale=payout_std,
            size=active_users,
        )
        payout_multiplier = np.clip(payout_multiplier, 0.0, 10.0)

        user_cash_winnings = spent_face_value * payout_multiplier
        withdrawn_winnings = user_cash_winnings * withdrawal_rate

        # If part of winnings is not withdrawn, assume it can create extra turnover and GGR.
        recycled_winnings = user_cash_winnings * (1.0 - withdrawal_rate)
        recycled_turnover = recycled_winnings * recycled_winnings_turnover_multiplier
        recycled_ggr = recycled_turnover * ggr_margin

        realized_uplift = rng.normal(
            loc=expected_uplift_mean,
            scale=expected_uplift_std,
            size=active_users,
        )
        realized_uplift = np.clip(realized_uplift, 0.0, 5.0)

        realized_response = rng.normal(loc=response_mean, scale=response_std, size=active_users)
        realized_response = np.clip(realized_response, 0.0, 1.5)

        # The more of the granted freebet is used, the stronger the short-term engagement effect.
        engagement_factor = 0.50 + 0.50 * utilization

        incremental_ggr_from_engagement = (
            positive_ggr
            * realized_uplift
            * realized_response
            * engagement_factor
            * offered.astype(float)
        )

        total_incremental_ggr = incremental_ggr_from_engagement + recycled_ggr

        granted_total = float(grant.sum())
        spent_total = float(spent_face_value.sum())
        burned_total = float(burned_face_value.sum())
        cash_payout_total = float(user_cash_winnings.sum())
        withdrawn_total = float(withdrawn_winnings.sum())
        recycled_ggr_total = float(recycled_ggr.sum())
        incremental_ggr_total = float(total_incremental_ggr.sum())

        cash_cost = withdrawn_total
        accounting_bonus_cost = spent_total

        net_incremental_profit_cash = incremental_ggr_total - cash_cost
        net_incremental_profit_accounting = incremental_ggr_total - accounting_bonus_cost

        ggr_after_cash_cost = ggr + incremental_ggr_total - cash_cost
        ggr_after_accounting_cost = ggr + incremental_ggr_total - accounting_bonus_cost

        users_offered = int(offered.sum())
        users_no_action = int(active_users - users_offered)
        users_used_freebet = int((spent_face_value > 0).sum())

        summary_rows.append(
            {
                "simulation": sim_id,
                "budget_ceiling": budget_ceiling,
                "granted_freebet_face_value": granted_total,
                "spent_freebet_face_value": spent_total,
                "burned_or_unused_freebet": burned_total,
                "budget_not_used": budget_ceiling - granted_total,
                "budget_utilization_rate": safe_div(granted_total, budget_ceiling, 0.0),
                "freebet_utilization_rate": safe_div(spent_total, granted_total, 0.0),
                "cash_payout_from_freebets": cash_payout_total,
                "withdrawn_winnings": withdrawn_total,
                "recycled_ggr_from_unwithdrawn_winnings": recycled_ggr_total,
                "incremental_ggr": incremental_ggr_total,
                "incremental_ggr_pct": safe_div(incremental_ggr_total, ggr, 0.0),
                "cash_cost": cash_cost,
                "accounting_bonus_cost": accounting_bonus_cost,
                "net_incremental_profit_cash": net_incremental_profit_cash,
                "net_incremental_profit_accounting": net_incremental_profit_accounting,
                "roi_cash_basis": safe_div(net_incremental_profit_cash, cash_cost, np.nan),
                "roi_accounting_basis": safe_div(
                    net_incremental_profit_accounting,
                    accounting_bonus_cost,
                    np.nan,
                ),
                "ggr_after_cash_cost": ggr_after_cash_cost,
                "ggr_after_accounting_cost": ggr_after_accounting_cost,
                "ggr_change_after_cash_cost_pct": safe_div(ggr_after_cash_cost - ggr, ggr, 0.0),
                "users_offered": users_offered,
                "users_no_action": users_no_action,
                "users_used_freebet": users_used_freebet,
                "offer_rate": safe_div(users_offered, active_users, 0.0),
                "use_rate_among_offered": safe_div(users_used_freebet, users_offered, 0.0),
                "candidate_users": int(candidate.sum()),
                "avg_grant_per_offered_user": safe_div(granted_total, users_offered, 0.0),
                "median_grant_all_users": float(np.median(grant)),
                "p90_grant_all_users": float(np.percentile(grant, 90)),
                "p95_grant_all_users": float(np.percentile(grant, 95)),
                "p99_grant_all_users": float(np.percentile(grant, 99)),
            }
        )

        sample_size = min(sample_per_sim, active_users)
        sample_idx = rng.choice(active_users, size=sample_size, replace=False)

        sampled_rows.append(
            pd.DataFrame(
                {
                    "simulation": sim_id,
                    "previous_user_ggr": user_ggr_prev[sample_idx],
                    "offered": offered[sample_idx],
                    "freebet_granted": grant[sample_idx],
                    "freebet_spent": spent_face_value[sample_idx],
                    "freebet_burned_or_unused": burned_face_value[sample_idx],
                    "utilization_rate": utilization[sample_idx],
                    "cash_payout_from_freebet": user_cash_winnings[sample_idx],
                    "withdrawn_winnings": withdrawn_winnings[sample_idx],
                    "incremental_ggr": total_incremental_ggr[sample_idx],
                    "expected_incremental_ggr": expected_incremental_ggr[sample_idx],
                    "net_incremental_profit_cash": total_incremental_ggr[sample_idx]
                    - withdrawn_winnings[sample_idx],
                }
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    sample_df = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else pd.DataFrame()

    metadata = {
        "expected_payout_multiplier": expected_payout_multiplier,
        "expected_cash_cost_per_granted_eur": expected_cash_cost_per_granted_eur,
        "ggr_margin": ggr_margin,
        "avg_ggr_per_user": avg_ggr_per_user,
    }

    return summary_df, sample_df, metadata


# -------------------------------------------------
# UI
# -------------------------------------------------

st.title("AOR: bookmaker-realistic модель фрибетов")
st.caption(
    "Monte Carlo-модель на 30 дней: 20% GGR используется как верхний лимит бюджета, "
    "Brain выбирает OFFER или NO_ACTION, а результат считается через incremental GGR, cash cost и ROI."
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
    st.header("Бюджет и decisioning")

    simulations = st.slider(
        "Количество симуляций",
        min_value=1,
        max_value=300,
        value=300,
        step=1,
    )

    allocation_pool_pct = st.slider(
        "Верхний лимит пула фрибетов от GGR",
        min_value=0.00,
        max_value=0.50,
        value=0.20,
        step=0.01,
        format="%.2f",
        help="0.20 означает, что 20% GGR прошлого периода — это потолок бюджета, а не обязательная сумма к трате.",
    )

    force_spend_budget = st.toggle(
        "Принудительно распределять максимум бюджета",
        value=False,
        help="В реальной модели лучше выключить: Brain не должен тратить бюджет, если expected profit недостаточный.",
    )

    min_freebet_per_user = st.number_input(
        "Минимальный фрибет на пользователя, €",
        min_value=0.0,
        value=0.0,
        step=0.5,
        format="%.2f",
    )

    max_freebet_per_user = st.number_input(
        "Максимальный фрибет на пользователя, €",
        min_value=0.1,
        value=25.0,
        step=1.0,
        format="%.2f",
    )

    max_cash_cost_to_incremental_ggr = st.slider(
        "Макс. cash cost / expected incremental GGR",
        min_value=0.05,
        max_value=1.50,
        value=0.70,
        step=0.05,
        help="Защита ROI: ожидаемый cash cost фрибета не должен превышать заданную долю expected incremental GGR.",
    )

    min_expected_profit_per_user = st.number_input(
        "Минимальная expected profit на пользователя, €",
        min_value=-100.0,
        value=0.0,
        step=0.5,
        format="%.2f",
    )

    restricted_share = st.slider(
        "Restricted / RG-risk users",
        min_value=0.00,
        max_value=0.30,
        value=0.02,
        step=0.01,
        format="%.2f",
    )

    bonus_abuse_share = st.slider(
        "Bonus-abuse / excluded users",
        min_value=0.00,
        max_value=0.30,
        value=0.03,
        step=0.01,
        format="%.2f",
    )

    st.divider()
    st.header("User-level GGR")

    ggr_distribution_mode = st.selectbox(
        "Распределение user-level GGR",
        [
            "Skewed bookmaker tail (recommended)",
            "Gaussian / Normal (original task)",
        ],
        index=0,
        help="Gaussian оставлен для первого ТЗ. Skewed tail ближе к реальному букмекеру.",
    )

    gaussian_ggr_cv = st.slider(
        "Gaussian CV для user-level GGR",
        min_value=0.05,
        max_value=5.00,
        value=1.50,
        step=0.05,
    )

    lognormal_sigma = st.slider(
        "Сила long-tail распределения",
        min_value=0.10,
        max_value=3.00,
        value=1.35,
        step=0.05,
        help="Чем выше значение, тем больше GGR концентрируется в малой доле пользователей.",
    )

    negative_user_share = st.slider(
        "Доля negative-GGR users",
        min_value=0.00,
        max_value=0.50,
        value=0.08,
        step=0.01,
        format="%.2f",
    )

    negative_loss_pool_pct = st.slider(
        "Negative-GGR pool как % от GGR",
        min_value=0.00,
        max_value=0.50,
        value=0.07,
        step=0.01,
        format="%.2f",
        help="Например, 0.07 означает, что минусовые пользователи создают отрицательный GGR, равный 7% от общего GGR.",
    )

    allocation_noise_std = st.slider(
        "Gaussian noise в начислении фрибетов",
        min_value=0.00,
        max_value=2.00,
        value=0.35,
        step=0.05,
        help="Начисление остаётся пропорциональным GGR, но с шумом, имитирующим скоринг Brain.",
    )

    st.divider()
    st.header("Трата фрибетов")

    utilization_mean = st.slider(
        "Средняя доля использованных фрибетов",
        min_value=0.00,
        max_value=1.00,
        value=0.65,
        step=0.01,
    )

    utilization_std = st.slider(
        "Gaussian разброс использования",
        min_value=0.00,
        max_value=0.50,
        value=0.20,
        step=0.01,
    )

    avg_freebet_odds = st.slider(
        "Средний коэффициент ставки на фрибет",
        min_value=1.01,
        max_value=10.00,
        value=2.00,
        step=0.01,
    )

    freebet_margin = st.slider(
        "Маржа на ставках с фрибетом",
        min_value=0.00,
        max_value=0.30,
        value=0.10,
        step=0.01,
    )

    stake_returned = st.toggle(
        "Freebet stake returned",
        value=False,
        help="Обычно freebet работает как stake not returned. Если включить, cash cost сильно вырастет.",
    )

    payout_std = st.slider(
        "Волатильность cash payout с фрибетов",
        min_value=0.00,
        max_value=2.00,
        value=0.20,
        step=0.01,
    )

    withdrawal_rate = st.slider(
        "Доля выигрыша, которую пользователь забирает",
        min_value=0.00,
        max_value=1.00,
        value=1.00,
        step=0.01,
    )

    recycled_winnings_turnover_multiplier = st.slider(
        "Оборот с невыведенного выигрыша, x",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        help="Если withdrawal rate меньше 100%, остаток выигрыша может вернуться в оборот.",
    )

    st.divider()
    st.header("Incremental GGR")

    expected_uplift_mean = st.slider(
        "Средний uplift GGR от offer",
        min_value=0.00,
        max_value=2.00,
        value=0.18,
        step=0.01,
        format="%.2f",
        help="0.18 означает +18% к предыдущему user-level GGR у пользователя, которому дали offer.",
    )

    expected_uplift_std = st.slider(
        "Разброс uplift GGR",
        min_value=0.00,
        max_value=1.00,
        value=0.10,
        step=0.01,
    )

    response_mean = st.slider(
        "Средняя response probability / intensity",
        min_value=0.00,
        max_value=1.00,
        value=0.60,
        step=0.01,
    )

    response_std = st.slider(
        "Разброс response",
        min_value=0.00,
        max_value=0.50,
        value=0.15,
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
    )

    random_seed = st.number_input(
        "Random seed",
        min_value=1,
        value=42,
        step=1,
        format="%d",
    )


# -------------------------------------------------
# Derived inputs
# -------------------------------------------------

avg_stake = safe_div(turnover, bets_count, 0.0)
ggr_margin = safe_div(ggr, turnover, 0.0)
turnover_per_user = safe_div(turnover, active_users, 0.0)
ggr_per_user = safe_div(ggr, active_users, 0.0)
budget_ceiling = ggr * allocation_pool_pct

input_cols = st.columns(5)
input_cols[0].metric("Средняя ставка", format_eur(avg_stake, 2))
input_cols[1].metric("GGR margin", format_pct(ggr_margin, 2))
input_cols[2].metric("Оборот / user", format_eur(turnover_per_user, 2))
input_cols[3].metric("GGR / user", format_eur(ggr_per_user, 2))
input_cols[4].metric("Потолок бюджета", format_eur(budget_ceiling, 0))

with st.expander("Логика доработанной модели", expanded=False):
    st.markdown(
        """
**Главное изменение:** 20% GGR теперь считается не суммой, которую обязательно надо потратить, а **верхним лимитом бонусного бюджета**.

**1. Synthetic user-level GGR**

Если нет реальных данных по пользователям, модель создаёт синтетическое user-level распределение.  
Можно выбрать режим:

- `Skewed bookmaker tail` — более реалистично для букмекера: длинный хвост, VIP/high-value users и часть negative-GGR users.
- `Gaussian / Normal` — вариант из первого ТЗ.

**2. Начисление фрибета**

Фрибет распределяется пропорционально положительному user-level GGR, но только среди пользователей, прошедших eligibility:

`candidate = positive GGR user AND not restricted AND not bonus-abuse`

Далее Brain применяет `NO_ACTION`, если expected profit недостаточна.

**3. Защита ROI**

На уровне пользователя:

`Expected Cash Costᵢ = Grantᵢ × Expected Utilization × Expected Payout Multiplier × Withdrawal Rate`

`Expected Incremental Profitᵢ = Expected Incremental GGRᵢ - Expected Cash Costᵢ`

Offer выдаётся только если:

`Expected Incremental Profitᵢ >= minimum expected profit`

И дополнительно:

`Expected Cash Costᵢ <= Expected Incremental GGRᵢ × allowed cost ratio`

**4. Cash cost и bonus nominal разделены**

В отчёте отдельно считаются:

- `granted freebet face value` — сколько фрибетов начислили номинально;
- `spent freebet face value` — сколько фрибетов реально использовали;
- `cash payout from freebets` — выигрыш пользователя с фрибета;
- `withdrawn winnings` — cash cost для оператора;
- `incremental GGR` — дополнительный GGR от AOR;
- `net incremental profit` — дополнительный GGR минус cash cost.

**5. Freebet stake not returned**

По умолчанию модель считает, что ставка фрибета не возвращается:

`Expected Payout Multiplier = (1 - margin) × (odds - 1) / odds`
        """
    )

with st.spinner("Считаю Monte Carlo-симуляции..."):
    summary_df, sample_df, metadata = run_simulation(
        bets_count=int(bets_count),
        turnover=float(turnover),
        ggr=float(ggr),
        active_users=int(active_users),
        simulations=int(simulations),
        ggr_distribution_mode=ggr_distribution_mode,
        allocation_pool_pct=float(allocation_pool_pct),
        gaussian_ggr_cv=float(gaussian_ggr_cv),
        lognormal_sigma=float(lognormal_sigma),
        negative_user_share=float(negative_user_share),
        negative_loss_pool_pct=float(negative_loss_pool_pct),
        allocation_noise_std=float(allocation_noise_std),
        utilization_mean=float(utilization_mean),
        utilization_std=float(utilization_std),
        avg_freebet_odds=float(avg_freebet_odds),
        freebet_margin=float(freebet_margin),
        stake_returned=bool(stake_returned),
        payout_std=float(payout_std),
        withdrawal_rate=float(withdrawal_rate),
        recycled_winnings_turnover_multiplier=float(recycled_winnings_turnover_multiplier),
        expected_uplift_mean=float(expected_uplift_mean),
        expected_uplift_std=float(expected_uplift_std),
        response_mean=float(response_mean),
        response_std=float(response_std),
        max_cash_cost_to_incremental_ggr=float(max_cash_cost_to_incremental_ggr),
        min_expected_profit_per_user=float(min_expected_profit_per_user),
        min_freebet_per_user=float(min_freebet_per_user),
        max_freebet_per_user=float(max_freebet_per_user),
        restricted_share=float(restricted_share),
        bonus_abuse_share=float(bonus_abuse_share),
        force_spend_budget=bool(force_spend_budget),
        random_seed=int(random_seed),
        sample_per_sim=int(sample_per_sim),
    )

median_row = summary_df.median(numeric_only=True)

st.subheader("Результаты симуляции, median по всем прогонам")

result_cols = st.columns(5)
result_cols[0].metric("Потолок бюджета", format_eur(median_row["budget_ceiling"]))
result_cols[1].metric("Начислено фрибетов", format_eur(median_row["granted_freebet_face_value"]))
result_cols[2].metric("Потрачено фрибетов", format_eur(median_row["spent_freebet_face_value"]))
result_cols[3].metric("Cash cost", format_eur(median_row["cash_cost"]))
result_cols[4].metric("Неиспользованный бюджет", format_eur(median_row["budget_not_used"]))

result_cols_2 = st.columns(5)
result_cols_2[0].metric("Incremental GGR", format_eur(median_row["incremental_ggr"]))
result_cols_2[1].metric("Изменение GGR", format_pct(median_row["incremental_ggr_pct"], 2))
result_cols_2[2].metric("Net profit, cash basis", format_eur(median_row["net_incremental_profit_cash"]))
result_cols_2[3].metric("ROI, cash basis", format_pct(median_row["roi_cash_basis"], 1))
result_cols_2[4].metric("GGR after cash cost", format_eur(median_row["ggr_after_cash_cost"]))

result_cols_3 = st.columns(5)
result_cols_3[0].metric("Users with offer", format_num(median_row["users_offered"]))
result_cols_3[1].metric("NO_ACTION", format_num(median_row["users_no_action"]))
result_cols_3[2].metric("Offer rate", format_pct(median_row["offer_rate"], 1))
result_cols_3[3].metric("Use rate among offered", format_pct(median_row["use_rate_among_offered"], 1))
result_cols_3[4].metric("Avg grant / offered user", format_eur(median_row["avg_grant_per_offered_user"], 2))

if median_row["net_incremental_profit_cash"] < 0:
    st.error(
        "Модель показывает отрицательную median net incremental profit на cash basis. "
        "Нужно уменьшить размер фрибетов, поднять expected uplift, снизить offer rate или ужесточить ROI threshold."
    )
else:
    st.success(
        "Median net incremental profit на cash basis положительная: при текущих параметрах AOR проходит базовую экономическую проверку."
    )

st.info(
    f"Expected cash payout multiplier: {metadata['expected_payout_multiplier']:.3f}. "
    f"Expected cash cost на €1 начисленного фрибета: {metadata['expected_cash_cost_per_granted_eur']:.3f}. "
    "Cash cost считается отдельно от номинала фрибета, потому что freebet stake обычно не является прямым денежным оттоком."
)

# -------------------------------------------------
# Charts and tables
# -------------------------------------------------

tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(
    [
        "Распределение фрибетов",
        "Экономика AOR",
        "Decisioning",
        "Percentiles",
        "Данные",
    ]
)

with tab_1:
    st.markdown("#### Распределение начисленных фрибетов среди пользователей с offer")

    offered_sample = sample_df[sample_df["freebet_granted"] > 0].copy()

    if offered_sample.empty:
        st.warning("В текущих параметрах модель не выдаёт offer пользователям. Смягчите ROI threshold или увеличьте expected uplift.")
    else:
        fig_grant = px.histogram(
            offered_sample,
            x="freebet_granted",
            nbins=80,
            marginal="box",
            labels={"freebet_granted": "Начисленный фрибет, €"},
            title="Freebet granted, только users with offer",
        )
        fig_grant.update_layout(bargap=0.02)
        st.plotly_chart(fig_grant, use_container_width=True)

        fig_spent = px.histogram(
            offered_sample,
            x="freebet_spent",
            nbins=80,
            marginal="box",
            labels={"freebet_spent": "Потраченный фрибет, €"},
            title="Freebet spent, только users with offer",
        )
        fig_spent.update_layout(bargap=0.02)
        st.plotly_chart(fig_spent, use_container_width=True)

    st.markdown("#### Распределение cash payout с фрибетов")
    payout_sample = sample_df[sample_df["cash_payout_from_freebet"] > 0].copy()

    if payout_sample.empty:
        st.warning("Нет cash payout по текущим параметрам.")
    else:
        fig_payout = px.histogram(
            payout_sample,
            x="cash_payout_from_freebet",
            nbins=80,
            marginal="box",
            labels={"cash_payout_from_freebet": "Cash payout, €"},
            title="Cash payout from freebets",
        )
        fig_payout.update_layout(bargap=0.02)
        st.plotly_chart(fig_payout, use_container_width=True)

with tab_2:
    st.markdown("#### Распределение incremental profit по симуляциям")

    fig_profit = px.histogram(
        summary_df,
        x="net_incremental_profit_cash",
        nbins=min(60, max(10, simulations)),
        marginal="box",
        labels={"net_incremental_profit_cash": "Net incremental profit, €"},
        title="Net incremental profit, cash basis",
    )
    fig_profit.update_layout(bargap=0.02)
    st.plotly_chart(fig_profit, use_container_width=True)

    st.markdown("#### Incremental GGR vs cash cost")

    fig_scatter = px.scatter(
        summary_df,
        x="cash_cost",
        y="incremental_ggr",
        size="granted_freebet_face_value",
        hover_data=[
            "simulation",
            "net_incremental_profit_cash",
            "roi_cash_basis",
            "users_offered",
            "budget_utilization_rate",
        ],
        labels={
            "cash_cost": "Cash cost, €",
            "incremental_ggr": "Incremental GGR, €",
            "granted_freebet_face_value": "Granted freebet, €",
        },
        title="Если точка выше диагонали, incremental GGR выше cash cost",
    )

    min_axis = min(summary_df["cash_cost"].min(), summary_df["incremental_ggr"].min())
    max_axis = max(summary_df["cash_cost"].max(), summary_df["incremental_ggr"].max())

    fig_scatter.add_trace(
        go.Scatter(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            name="break-even",
            line=dict(dash="dash"),
        )
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Accounting view: bonus budget as cost")
    fig_accounting = px.histogram(
        summary_df,
        x="net_incremental_profit_accounting",
        nbins=min(60, max(10, simulations)),
        marginal="box",
        labels={"net_incremental_profit_accounting": "Net incremental profit, €"},
        title="Net incremental profit, accounting basis: incremental GGR - spent freebet face value",
    )
    fig_accounting.update_layout(bargap=0.02)
    st.plotly_chart(fig_accounting, use_container_width=True)

with tab_3:
    st.markdown("#### OFFER vs NO_ACTION")

    decision_df = pd.DataFrame(
        {
            "decision": ["OFFER", "NO_ACTION"],
            "users": [
                median_row["users_offered"],
                median_row["users_no_action"],
            ],
        }
    )

    fig_decision = px.bar(
        decision_df,
        x="decision",
        y="users",
        text="users",
        labels={"decision": "Decision", "users": "Пользователи"},
        title="Median users by decision",
    )
    fig_decision.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_decision, use_container_width=True)

    st.markdown("#### Offer rate по симуляциям")
    fig_offer_rate = px.line(
        summary_df,
        x="simulation",
        y="offer_rate",
        labels={"simulation": "Симуляция", "offer_rate": "Offer rate"},
        title="Доля пользователей, которым Brain выдал offer",
    )
    fig_offer_rate.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_offer_rate, use_container_width=True)

    st.markdown("#### Бюджет: потолок vs фактическое начисление")
    budget_line_df = summary_df[
        [
            "simulation",
            "budget_ceiling",
            "granted_freebet_face_value",
            "spent_freebet_face_value",
            "cash_cost",
        ]
    ].melt(id_vars="simulation", var_name="metric", value_name="value")

    fig_budget = px.line(
        budget_line_df,
        x="simulation",
        y="value",
        color="metric",
        labels={"simulation": "Симуляция", "value": "€", "metric": "Метрика"},
        title="Бюджет и фактическая нагрузка",
    )
    st.plotly_chart(fig_budget, use_container_width=True)

with tab_4:
    st.markdown("#### Percentiles по user-level sample")

    percentile_source = offered_sample if not offered_sample.empty else sample_df
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    percentile_df = pd.DataFrame(
        {
            "percentile": percentiles,
            "freebet_granted": [
                np.percentile(percentile_source["freebet_granted"], p)
                for p in percentiles
            ],
            "freebet_spent": [
                np.percentile(percentile_source["freebet_spent"], p)
                for p in percentiles
            ],
            "cash_payout_from_freebet": [
                np.percentile(percentile_source["cash_payout_from_freebet"], p)
                for p in percentiles
            ],
            "incremental_ggr": [
                np.percentile(percentile_source["incremental_ggr"], p)
                for p in percentiles
            ],
            "net_incremental_profit_cash": [
                np.percentile(percentile_source["net_incremental_profit_cash"], p)
                for p in percentiles
            ],
        }
    )

    st.dataframe(
        percentile_df.style.format(
            {
                "freebet_granted": "€{:,.2f}",
                "freebet_spent": "€{:,.2f}",
                "cash_payout_from_freebet": "€{:,.2f}",
                "incremental_ggr": "€{:,.2f}",
                "net_incremental_profit_cash": "€{:,.2f}",
            }
        ),
        use_container_width=True,
        height=360,
    )

    st.markdown("#### Percentiles по simulation-level results")

    simulation_percentile_metrics = [
        "granted_freebet_face_value",
        "spent_freebet_face_value",
        "cash_cost",
        "incremental_ggr",
        "net_incremental_profit_cash",
        "roi_cash_basis",
        "users_offered",
    ]

    sim_pct_rows = []
    for metric in simulation_percentile_metrics:
        values = summary_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue

        row = {"metric": metric}
        for p in [5, 25, 50, 75, 95]:
            row[f"p{p}"] = np.percentile(values, p)
        sim_pct_rows.append(row)

    sim_pct_df = pd.DataFrame(sim_pct_rows)

    st.dataframe(
        sim_pct_df,
        use_container_width=True,
        height=360,
    )

with tab_5:
    st.markdown("#### Summary по симуляциям")

    display_df = summary_df.copy()

    money_cols = [
        "budget_ceiling",
        "granted_freebet_face_value",
        "spent_freebet_face_value",
        "burned_or_unused_freebet",
        "budget_not_used",
        "cash_payout_from_freebets",
        "withdrawn_winnings",
        "recycled_ggr_from_unwithdrawn_winnings",
        "incremental_ggr",
        "cash_cost",
        "accounting_bonus_cost",
        "net_incremental_profit_cash",
        "net_incremental_profit_accounting",
        "ggr_after_cash_cost",
        "ggr_after_accounting_cost",
        "avg_grant_per_offered_user",
        "median_grant_all_users",
        "p90_grant_all_users",
        "p95_grant_all_users",
        "p99_grant_all_users",
    ]

    pct_cols = [
        "budget_utilization_rate",
        "freebet_utilization_rate",
        "incremental_ggr_pct",
        "roi_cash_basis",
        "roi_accounting_basis",
        "ggr_change_after_cash_cost_pct",
        "offer_rate",
        "use_rate_among_offered",
    ]

    formatters = {col: "€{:,.0f}" for col in money_cols if col in display_df.columns}
    formatters.update({col: "{:.2%}" for col in pct_cols if col in display_df.columns})

    st.dataframe(
        display_df.style.format(formatters),
        use_container_width=True,
        height=460,
    )

    csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать simulation summary CSV",
        data=csv,
        file_name="aor_bookmaker_realistic_simulation_summary.csv",
        mime="text/csv",
    )

    sample_csv = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать user-level sample CSV",
        data=sample_csv,
        file_name="aor_bookmaker_realistic_user_sample.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Production-рекомендация: заменить synthetic user-level GGR на реальные user-level данные: "
    "GGR, turnover, bet count, segment, VIP flag, churn score, bonus history, restrictions, jurisdiction, "
    "bonus abuse signals и фактический freebet redemption."
)
