import xgboost as xgb
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

def xg(x_train,y_train,x_test,y_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', 
                         n_estimators=100, 
                         learning_rate=0.1, 
                         max_depth=1000)
    
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R2 score: {r2}')
    return model

def parameter_searching(X_train,y_train,X_test,y_test,model):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [100, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2]
    }


    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    print(f'Best parameters: {grid_search.best_params_}')

    best_model = grid_search.best_estimator_
    y_pred_optimized = best_model.predict(X_test)
    mse_optimized = mean_squared_error(y_test, y_pred_optimized)
    r2_optimized = r2_score(y_test, y_pred_optimized)
    print(f'Optimized Mean Squared Error: {mse_optimized}')
    print(f'Optimized R2 score: {r2_optimized}')

def barrido_parametrico( X_train, X_test, y_train, y_test):
    feature_columns = X_train.columns
    results = []

    for i in range(1, len(feature_columns) + 1):
        selected_columns = feature_columns[:i]
        print(f"Entrenando con columnas: {list(selected_columns)}")
        X_train_subset = X_train[selected_columns]
        X_test_subset = X_test[selected_columns]
        model = xgb.XGBRegressor(objective='reg:squarederror', 
                                 n_estimators=100, 
                                 learning_rate=0.1, 
                                 max_depth=100,
                                 random_state=42)
        model.fit(X_train_subset, y_train)

        y_pred = model.predict(X_test_subset)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'num_features': i,
            'features': list(selected_columns),
            'mse': mse,
            'r2': r2
        })
        print(f"Resultados para {i} columnas: {list(selected_columns)}")
        print(f"MSE={mse:.4f}, R2={r2:.4f}\n")
