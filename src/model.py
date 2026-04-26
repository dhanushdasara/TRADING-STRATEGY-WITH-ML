from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(df):
    """
    Trains a Random Forest Classifier on the processed data.
    """
    # Define features: All columns starting with 'CDL_' (patterns) plus indicators
    features = [col for col in df.columns if 'CDL_' in col] + ['RSI', 'SMA_50', 'SMA_200']
    X = df[features]
    y = df['Target']
    
    # Split Data (Shuffle=False to respect time order)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Initialize and Train Model
    clf = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    predictions = clf.predict(X_test)
    
    print("--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions))
    
    # Return model and test data for backtesting
    return clf, X_test, y_test, predictions