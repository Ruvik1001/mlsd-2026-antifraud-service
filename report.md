# Отчёт: итоговый проект антифрод-сервиса (7_62)

Выполнил студент МАИ группы М80-205СВ-25 Руднев Виктор Владимирович

## Условия

| Параметр | Значение |
| --- | --- |
| ОС | Windows 10 (10.0.19045) |
| Python | 3.11.9 |
| sklearn | 1.7.2 |
| seed | 2026 |
| заказов | 200000 / 30 дней |
| fraud_rate | 0.005185 |
| data sha256 | 3ee920da787fd3117f08eefffeb4fc63e9bbf763b86a13a27daa5012f673fea0 |
| threshold | 0.9896 |
| model | DecisionTree depth=4 |

`pytest -q`: 8 passed ≈ 4.5 с. Полный прогон: `python run_all.py`.

## Лестница и тест

| модель | valid AUC | test AUC | cost ₽ | reviews | FN | FP |
| --- | --- | --- | --- | --- | --- | --- |
| rule amount/age | 0.50 | 0.50 | 1 352 000 | 0 | 169 | 0 |
| linear | 0.972 | 0.973 | 624 550 | 149 | 77 | 57 |
| tree | 0.965 | 0.958 | 503 750 | 376 | 58 | 265 |

Бутстрэп по дням Δcost(tree−rule): mean −556821, 95% CI [−848250, −281300].
Без проверок ущерб 1.35 млн ₽; с моделью 0.50 млн. Review share на тесте 1.12% < 1.5%.

Срезы: new accounts AUC 0.876 (n=568, pos=7) — слабее overall; high device_risk AUC 0.868.
Brier после калибровки на valid: 0.00415.

## Сервис и оптимизация

Контракт `/predict`: version, degraded, reasons. Неизвестный payment → review.
Timeout store 20 мс → orders_1h=0 + feature_timeout. Без модели — rule-v1, ready=503.

Оптимизация: batch64 `predict_proba` вместо поштучного.

| путь | p50 мс/строка | p95 мс/строка |
| --- | --- | --- |
| single | 0.061 | 0.061 |
| batch64 | 0.00099 | 0.00099 |

Качество то же. Это замер вызова модели, не всего HTTP. На машине замера; на другой — иначе.
Пик оформления: учебный ×5 к среднему; p95 HTTP на TestClient ≪ 100 мс.

## Выпуск и мониторинг

Shadow → canary 10% → откат по p95 canary (окно 59, p95≈143 мс > 100).
Журнал: `reports/release_journal.json`. После отката фаза `rolled_back`, трафик на v1.

Инцидент в monitor: залипший device_risk + boost amount → PSI(device_risk)≈0.53 → алерт
«investigate_data_before_quality». Зрелые метки считаются только при label_ready_ts ≤ asof.

## Переобучение

Один цикл на расширенном train: кандидат depth=5 принят (cost 150800 < 223150, AUC выше).
Предыдущие артефакты в `artifacts/previous/`. Владелец: oncall-ml.

## Границы

Симуляция меток через 7 дней — упрощение. Selection bias у reviewed в проде не снят,
только описан. A/B на живых покупателях не делался.
