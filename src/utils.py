import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(route,X,y,y_bins,name,test=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y_bins
    )
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv(f'{route}/{name}_train.csv')
    if test is None:
        test_data.to_csv(f'{route}/{name}_test.csv')
    else:
        test_data.to_csv(f'{route}/{name}_val.csv')

