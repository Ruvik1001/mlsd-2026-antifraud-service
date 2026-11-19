# Итоговый проект: антифрод заказов в сервисе

Выполнил студент МАИ группы М80-205СВ-25 Руднев Виктор Владимирович

## Задача

Решение: скоринг заказа при оформлении → approve / review.
Пользователь: покупатель маркетплейса; действие после прогноза — ручная проверка.
Главная метрика: стоимость ошибок `FN*8000 + FP*150` ₽.
Ограничения: ≤3000 review на 200000 заказов/мес; p95 ≤ 100 мс.

Пиковый профиль: вечерний всплеск ~5× среднего (учебный), p95 меряем на локальном стенде.

## Данные

Генератор `data/generate.py`, seed 2026.
Признаки: amount_rub, account_age_days, orders_1h, device_risk, payment_method.
Скрытое: `p_fraud_hidden` (только проверка).
Метка готова через 7 суток после `event_ts`.
Доля нарушений на обучающем периоде калибруется к ~0.5%.

## Запуск

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# быстрый контур (~20k)
python run_all.py --quick

# полный месяц (200k)
python run_all.py

# сервис
uvicorn src.service:app --port 8080
```

Быстрые тесты: `pytest -q`
Полный эксперимент пишет отчёты в `reports/` и артефакты в `artifacts/`.

## Обучение

Границы дней: train 0–14, valid 14–18, test 18–23, replay 23–30 (`configs/train.json`).
Лестница: правило amount/age → LogisticRegression → DecisionTree(depth=4).
Дерево выбрано за взаимодействие `orders_1h * device_risk`.
Конфиг, seed, sha256 данных — в `artifacts/manifest.json`.

## Оценка

См. `reports/offline_eval.json`: AUC/стоимость правила и кандидата, бутстрэп Δcost по дням,
срезы (новые аккаунты, высокий device_risk), калибровка (Brier).

## Производительность

`reports/benchmark.json`: single vs batch64 predict на той же модели.
Оптимизация — пакетный `predict_proba`; качество не меняется.
Цифры: «на машине замера; на другой — иначе».

## Выпуск

Тень → canary 10% → откат при error_rate>1% в двух окнах подряд
или p95>100 мс. Журнал: `reports/release_journal.json`, сводка `replay_summary.json`.

## Эксплуатация

Немедленные сигналы: PSI по признакам (`reports/monitor.json`).
Качество — только после зрелости меток.
Инцидент: залипший device_risk + сдвиг amount → PSI-алерт → сначала данные.
Переобучение: `src/retrain.py`, решение в `reports/retrain.json`; при отказе остаётся текущая модель.

## Ограничения

Симуляция потока и меток учебная. Реальный A/B и задержки подтверждения фрода
не воспроизведены. Selection bias меток у reviewed-заказов описан в monitor-отчёте.
