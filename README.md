# AOR — симуляционная модель с расчётом Coins и фрибетов

Русская версия Streamlit-дашборда для Art of Retention.

## Главное изменение

Coins и фрибеты больше не вводятся вручную.

Модель сама рассчитывает максимальное и плановое количество Coins / фрибетов на основе дополнительной прибыли от применения программы.

Главное ограничение:

```text
Стоимость Coins / фрибетов не может превышать Incremental Gross GGR
```

## Фиксированное правило текущей версии

```text
1 Coin = 1 фрибет = €1 бонусного бюджета
```

## Как считается

```text
Max Reward Budget = max(0, Incremental Gross GGR)

Planned Reward Budget = Max Reward Budget × Reward Budget Share

Coins = floor(Planned Reward Budget / €1)

Freebets = Coins

Program Cost = Coins × €1

Net Incremental GGR = Incremental Gross GGR - Program Cost

ROI = Net Incremental GGR / Program Cost
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
