# AOR — симуляционная модель на 30 дней

Русская версия Streamlit-дашборда для проекта Art of Retention.

## Что делает модель

Дашборд позволяет вводить усреднённые данные букмекера:

- количество ставок за период;
- оборот;
- GGR;
- количество активных пользователей.

После этого модель запускает минимум 200 симуляций на период 30 календарных дней, распределяет ставки между пользователями по усечённой модели Гаусса и рассчитывает:

- среднее количество фрибетов на active user;
- стоимость программы;
- дополнительный gross GGR;
- net incremental GGR;
- ROI программы;
- долю прибыльных симуляций;
- break-even;
- распределения ROI, стоимости программы и фрибетов на пользователя.

## Запуск локально

```bash
pip install -r requirements.txt
streamlit run aor_retention_dashboard_ru.py
```

## Ключевые формулы

```text
average_stake = turnover / bets
hold = GGR / turnover
GGR_per_bet = GGR / bets

total_freebets = completed_users * missions_per_completed_user * coins_per_completed_mission

program_cost = total_freebets * freebet_value_eur * redemption_rate * freebet_cost_factor

incremental_ggr_gross = AOR gross GGR - baseline GGR
incremental_ggr_net = incremental_ggr_gross - program_cost

ROI = incremental_ggr_net / program_cost
```

В этой версии модели используется курс:

```text
1 Coin = 1 фрибет
```
