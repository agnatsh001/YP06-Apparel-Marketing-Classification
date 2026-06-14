# Классификация клиентов интернет-магазина одежды

[![Python](https://img.shields.io/badge/Python-3.11.15-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow)](https://catboost.ai/)

---

## Описание задачи

Интернет-магазин проводит рекламные рассылки и планирует будущие продажи. Цель проекта — построить модель бинарной классификации, которая предсказывает вероятность покупки клиентом в течение 90 дней. Модель позволяет выделить пользователей, готовых совершить покупку в ближайшее время, и использовать эту информацию для планирования продаж и оптимизации бизнес-процессов.

**Тип задачи:** бинарная классификация (дисбаланс классов ~1:51)  
**Метрика:** ROC-AUC — устойчива к дисбалансу и измеряет способность модели ранжировать клиентов по вероятности покупки

---

## Данные

| Файл | Описание |
|---|---|
| `apparel-purchases.csv` | История покупок клиентов |
| `apparel-messages.csv` | История рекламных рассылок (1 348 MB, 12.7M строк) |
| `apparel-target_binary.csv` | Целевая переменная (1 — покупка в течение 90 дней) |
| `full_campaign_daily_event.csv` | Ежедневная статистика кампаний |
| `full_campaign_daily_event_channel.csv` | Статистика кампаний по каналам |

> ⚠️ Данные не включены в репозиторий. После клонирования создайте папку `data/` в корне проекта и поместите в неё все файлы вручную:
> ```
> mkdir data
> # скопируйте файлы в папку data/
> ```

**Объём выборки:** 49 849 клиентов, из которых 961 (1.93%) совершили целевое действие.

**Временны́е границы:**
- `CUTOFF = 2024-02-16` — признаки строятся по событиям **до** этой даты
- Целевая переменная — покупка в окне `2024-02-16 → 2024-05-16`

---

## Инжиниринг признаков

Из пяти источников данных построено **37 признаков** пяти групп:

| Группа | Признаков | Примеры |
|---|---:|---|
| RFM и покупки | 8 | `recency_purchase`, `monetary_total`, `n_orders` |
| Ценовые | 3 | `price_q75`, `avg_check` |
| Товарные категории | 7 | `n_unique_categories`, `has_root_4` |
| Вовлечённость в рассылки | 15 | `open_rate`, `recency_last_click`, `push_share` |
| Качество кампаний | 4 | `avg_camp_open_rate`, `avg_camp_ctr` |

`apparel-messages.csv` обработан через DuckDB без загрузки в память.

---

## Результаты

| Модель | CV ROC-AUC | Std | Test ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression (baseline) | 0.7567 | ±0.017 | — |
| LightGBM дефолт | 0.6836 | ±0.026 | — |
| LightGBM + Optuna | 0.7652 | ±0.017 | — |
| CatBoost + Optuna | 0.7702 | ±0.016 | — |
| **Стекинг (финал)** | **0.7728** | **±0.017** | **0.7810** |

**Финальная архитектура:** LogReg + LightGBM + CatBoost → мета-LogReg

**Калибровка вероятностей** (IsotonicRegression на OOF):

- До калибровки: `sum(predict_proba)` = 3 947 при фактических 192 покупателях (расхождение ×21)
- После калибровки: прогноз 186 покупателей, отклонение −2.9%
- ROC-AUC: 0.7810 → 0.7809 (ранжирование сохраняется с небольшим отклонением)

Откалиброванные вероятности позволяют использовать `sum(predict_proba)` как оценку ожидаемого числа покупателей в следующие 90 дней — основу для планирования продаж и запасов.

---

## Структура репозитория

```
YP06_Apparel-Marketing-Classification/
├── data/                        # данные (в .gitignore)
├── notebooks/                   # вспомогательные тетради (если есть)
├── YP06_notebook.ipynb          # основной ноутбук
├── environment.yml              # conda-окружение
├── requirements.txt             # зависимости для pip
├── README.md
└── .gitignore
```

---

## Установка окружения

```bash
conda env create -f environment.yml
conda activate yandex_practicum
```

---

## Запуск

```bash
conda activate yandex_practicum
jupyter notebook YP06_notebook.ipynb
```

> ⏱ Время полного прогона: **~30 минут** (Apple M1 Pro, 16 GB RAM)

---

## Стек

| Категория | Инструменты |
|---|---|
| Язык | Python 3.11.15 |
| Данные | pandas 3.0.3, DuckDB 1.5.3 (большие файлы) |
| ML | scikit-learn 1.8.0, LightGBM 4.6.0, CatBoost 1.2.10 |
| Подбор гиперпараметров | Optuna 4.9.0 (TPESampler, seed=42) |
| Корреляционный анализ | phik 0.12.5 |
| Визуализация | matplotlib, seaborn |
| Среда | Jupyter Notebook, conda |

---

## Автор

Наталья Шишкина — [agnatsh](https://github.com/agnatsh001)
