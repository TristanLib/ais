| Split | Model | Status | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | Train | Val | Test |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| temporal_test | constant_acceleration | ok | 35076.6 | 76360.6 | 737164.2 | 28836.3 | 130428 | 27948 | 27950 |
| temporal_test | constant_velocity | ok | 2751.3 | 4469.8 | 77190.7 | 1742.6 | 130428 | 27948 | 27950 |
| temporal_test | gru_baseline | ok | 25215.5 | 25571.8 | 29923.9 | 14863.6 | 130428 | 27948 | 27950 |
| temporal_test | kalman_filter_cv | ok | 1759.7 | 2704.5 | 22422.7 | 1101.1 | 130428 | 27948 | 27950 |
| temporal_test | linear_lstsq | ok | 3052.3 | 4908.7 | 13349.1 | 1910.3 | 130428 | 27948 | 27950 |
| temporal_test | lstm_baseline | ok | 36039.2 | 36116.8 | 57826.3 | 22176.2 | 130428 | 27948 | 27950 |
| temporal_test | ridge_lstsq | ok | 3141.7 | 5079.5 | 12468.9 | 1965.8 | 130428 | 27948 | 27950 |
| temporal_test | tcn_baseline | ok | 47095.5 | 47078.0 | 48378.2 | 29703.9 | 130428 | 27948 | 27950 |
| temporal_test | transformer_baseline | ok | 56310.7 | 55923.5 | 57766.4 | 34854.0 | 130428 | 27948 | 27950 |
| vessel_disjoint_test | constant_acceleration | ok | 36237.1 | 67988.2 | 1786906.6 | 55853.5 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | constant_velocity | ok | 9553.5 | 17014.2 | 198051.0 | 6181.8 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | gru_baseline | ok | 23989.0 | 24569.4 | 27376.0 | 14911.1 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | kalman_filter_cv | ok | 3109.4 | 5979.6 | 47635.6 | 1941.5 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | linear_lstsq | ok | 8113.3 | 14869.8 | 172502.5 | 5291.8 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | lstm_baseline | ok | 51010.6 | 50890.1 | 77725.3 | 32747.2 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | ridge_lstsq | ok | 3446.9 | 6463.4 | 18536.5 | 2153.1 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | tcn_baseline | ok | 32833.1 | 34379.1 | 34775.3 | 20214.3 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | transformer_baseline | ok | 47559.1 | 45907.0 | 46592.6 | 30158.9 | 130018 | 28597 | 27711 |
