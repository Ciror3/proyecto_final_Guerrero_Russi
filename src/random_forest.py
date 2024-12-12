import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rf(X_train, X_test, y_train, y_test):
    rf_model = RandomForestRegressor(
        n_estimators=2,      
        max_depth=2,        
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    evaluate_model(y_train,y_pred_train)
    evaluate_model(y_test,y_pred_test)
    return rf_model


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2: {r2:.4f}")
    return rmse, r2

    
