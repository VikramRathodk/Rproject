from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_lasso_regression_model(df, alpha_value):
    features = ['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']
    target = 'weighted_popularity_score'

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

    # Initialize and train a Lasso Regression model with an alpha value for regularization
    lasso_model = Lasso(alpha=alpha_value)  # Alpha is the regularization strength
    lasso_model.fit(X_train, y_train)

    # Make predictions
    predictions = lasso_model.predict(X_test)

    # Calculate Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, and R-squared
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)  # Squared=False returns RMSE
    mae = mean_absolute_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)

    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)
    print("R-squared (Coefficient of Determination):", r_squared)

    return lasso_model

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras import regularizers
# from keras.optimizers import Adam  # Import Adam optimizer

# def train_neural_network_model(df):
#     features = ['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']
#     target = 'weighted_popularity_score'

#     X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Change model architecture by adding dropout and regularization
#     model = Sequential()
#     model.add(Dense(128, input_dim=len(features), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#     model.add(Dropout(0.3))  # Example dropout layer
#     model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#     model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#     model.add(Dense(1))

#     # Compile the model with the Adam optimizer
#     model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mae'])

#     # Train the model
#     history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=0)

#     # Evaluate the model
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     mae = history.history['mae'][-1]
#     rmse = mse ** 0.5
#     r_squared = r2_score(y_test, predictions)

#     print("Mean Squared Error:", mse)
#     print("Mean Absolute Error:", mae)
#     print("Root Mean Squared Error:", rmse)
#     print("R-squared (Coefficient of Determination):", r_squared)

#     return model
