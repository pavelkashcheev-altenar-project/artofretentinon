# AOR — симуляционная модель с миссиями, Coins и фрибетами

Русская версия Streamlit-дашборда для Art of Retention.

## Что учитывает эта версия

1. **Пользователь может выполнить больше одной миссии в месяц.**

```text
missions_per_completed_user = 1 + Poisson(avg_missions_per_completed_user - 1)
```

Количество миссий ограничивается параметром:

```text
max_missions_per_completed_user
```

2. **У пользователей есть стартовые фрибеты в начале симуляции.**

```text
initial_freebets_total = initial_freebets_per_active_user × active_users
```

3. **Пользователи могут тратить фрибеты в течение симуляции.**

Каждый день списывается заданная доля доступного баланса:

```text
daily_freebet_spend_rate
```

Стартовые фрибеты и AOR-фрибеты считаются отдельно.

4. **Coins и AOR-фрибеты не вводятся вручную.**

Они рассчитываются из дополнительного gross GGR:

```text
Max Reward Budget = max(0, Incremental Gross GGR)

Planned Reward Budget = Max Reward Budget × Reward Budget Share

Coins = floor(Planned Reward Budget / €1)

AOR Freebets Issued = Coins
```

## Главное ограничение

```text
Стоимость выданных Coins / AOR-фрибетов не может превышать дополнительную прибыль от программы.
```

## Фиксированное правило текущей версии

```text
1 Coin = 1 фрибет = €1 бонусного бюджета
```

## Запуск

```bash
pip install -r requirements.txt
streamlit run aor_retention_dashboard_ru.py
```

## Файлы

```text
aor_retention_dashboard_ru.py
requirements.txt
README.md
```
