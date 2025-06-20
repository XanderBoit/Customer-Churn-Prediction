from sklearn.model_selection import train_test_split

def split_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=1
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=1
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test