# ReFine

├── __init__.py
├── data
│   ├── Merged-update_hourly.csv
│   ├── energy_weather.csv
│   ├── jena_climate_2009_2016.csv
│   ├── jena_climate_2009_2016_hourly.csv
│   └── pollution.csv
├── helper.py
├── model
│   ├── __init__.py
│   ├── cnn.py
│   └── mlp.py
├── postprocess
│   ├── load_postprocess.ipynb
│   ├── performance.py
│   ├── pm_postprocess.ipynb
│   ├── pressure_postprocess.ipynb
│   ├── price_postprocess.ipynb
├── preprocess.py
├── saved_models_hyper
├── saved_models_mlp
└── training_mlp
    ├── load
    │   ├── load_all.ipynb
    │   ├── load_all_ft.ipynb
    │   ├── load_all_weighted_EVT.ipynb
    │   ├── load_all_weighted_IPF.ipynb
    │   ├── load_all_weighted_META.ipynb
    ├── pm
    │   ├── pm_all.ipynb
    │   ├── pm_all_ft.ipynb
    │   ├── pm_all_weighted_EVT.ipynb
    │   ├── pm_all_weighted_IPF.ipynb
    │   ├── pm_all_weighted_META.ipynb
    ├── pressure
    │   ├── pressure_all.ipynb
    │   ├── pressure_all_ft.ipynb
    │   ├── pressure_all_weighted_EVT.ipynb
    │   ├── pressure_all_weighted_IPF.ipynb
    │   ├── pressure_all_weighted_META.ipynb
    ├── price
    │   ├── price_all.ipynb
    │   ├── price_all_ft.ipynb
    │   ├── price_all_weighted_EVT.ipynb
    │   ├── price_all_weighted_IPF.ipynb
    │   ├── price_all_weighted_META.ipynb
    └── water
        ├── ws_all.ipynb
        ├── ws_all_ft.ipynb
        ├── ws_all_old.ipynb
        ├── ws_all_weighted_EVT.ipynb
        ├── ws_all_weighted_IPF.ipynb
        └── ws_all_weighted_META.ipynb
