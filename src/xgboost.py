import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def xg(x_train,y_train,x_test,y_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', 
                         n_estimators=100, 
                         learning_rate=0.1, 
                         max_depth=100)
    
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

def parameter_searching(X_train,y_train,X_test,y_test,model):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Crear el modelo XGBoost

    # Realizar la búsqueda en cuadrícula
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    # Imprimir los mejores parámetros
    print(f'Best parameters: {grid_search.best_params_}')

    # Evaluar el modelo optimizado
    best_model = grid_search.best_estimator_
    y_pred_optimized = best_model.predict(X_test)
    mse_optimized = mean_squared_error(y_test, y_pred_optimized)
    print(f'Optimized Mean Squared Error: {mse_optimized}')