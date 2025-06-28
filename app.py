import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import io

# Set page configuration
st.set_page_config(layout="wide", page_title="Stock Price Predictor", page_icon="📈")

st.title("📈 Stock Price Prediction with LSTM")
st.markdown("Upload your historical stock data (CSV format) and predict future prices using an LSTM neural network.")

# --- File Uploader ---
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

data = None
prices = None

if uploaded_file is not None:
    # Read the uploaded CSV file
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(data.head())

        if 'Close' in data.columns:
            prices = data['Close'].values.reshape(-1, 1)
            st.success("Data loaded successfully!")
        else:
            st.error("Error: 'Close' column not found in your CSV. Please ensure your data has a 'Close' price column.")
            data = None # Reset data if 'Close' column is missing
    except Exception as e:
        st.error(f"Error reading file: {e}. Please ensure it's a valid CSV.")
        data = None # Reset data on error
else:
    st.info("Please upload a CSV file to begin. A 'Close' column is required.")

if data is not None and prices is not None:
    st.header("2. Configure Model Parameters")

    # --- Sidebar for Parameters ---
    st.sidebar.header("Model Hyperparameters")
    sequence_length = st.sidebar.slider("Sequence Length (Look-back days)", 5, 60, 10, 5)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50, 10)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, 16)
    lstm_units_1 = st.sidebar.slider("LSTM Units (Layer 1)", 30, 100, 50, 10)
    lstm_units_2 = st.sidebar.slider("LSTM Units (Layer 2)", 30, 100, 50, 10)
    split_ratio = st.sidebar.slider("Training Data Split Ratio", 0.6, 0.9, 0.8, 0.05)


    # --- Data Normalization ---
    st.subheader("Data Normalization")
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_normalized = scaler.fit_transform(prices)
    st.write("Prices normalized between 0 and 1 using MinMaxScaler.")

    # --- Create Sequences ---
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(prices_normalized, sequence_length)
    st.write(f"Created {len(X)} sequences with a look-back of {sequence_length} days.")

    # --- Split Data ---
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    st.write(f"Data split: {len(X_train)} samples for training, {len(X_test)} for testing.")

    # --- Build and Train Model ---
    st.header("3. Build and Train LSTM Model")
    if st.button("Train Model"):
        if len(X_train) == 0 or len(X_test) == 0:
            st.error("Not enough data to create training/testing sets. Please check your data or sequence length.")
        else:
            with st.spinner("Building and training model... This may take a while."):
                model = Sequential()
                model.add(LSTM(units=lstm_units_1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(LSTM(units=lstm_units_2))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Display model summary in an expander
                with st.expander("Model Summary"):
                    model.summary(print_fn=lambda x: st.text(x))

                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                st.success("Model training complete!")

                # Plot training loss
                st.subheader("Training Loss History")
                fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
                ax_loss.plot(history.history['loss'], label='Training Loss')
                ax_loss.set_title('Model Training Loss')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss (MSE)')
                ax_loss.legend()
                ax_loss.grid(True)
                st.pyplot(fig_loss)

                # --- Make Predictions ---
                st.header("4. Make Predictions and Evaluate")
                y_pred = model.predict(X_test)

                y_pred_original = scaler.inverse_transform(y_pred)
                y_test_original = scaler.inverse_transform(y_test)

                mse = mean_squared_error(y_test_original, y_pred_original)
                st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")

                # --- Visualize Predictions ---
                st.header("5. Visualization of Predictions")
                fig_pred, ax_pred = plt.subplots(figsize=(14, 7))
                ax_pred.plot(y_test_original, label='Actual Prices', color='blue', linewidth=2)
                ax_pred.plot(y_pred_original, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
                ax_pred.set_title('Stock Price Prediction with LSTM', fontsize=16)
                ax_pred.set_xlabel('Time Step (Days)', fontsize=12)
                ax_pred.set_ylabel('Stock Price', fontsize=12)
                ax_pred.legend(fontsize=10)
                ax_pred.grid(True, linestyle=':', alpha=0.7)
                st.pyplot(fig_pred)

                st.markdown("""
                ---
                **Notes:**
                * Stock price prediction is inherently difficult and past performance is not indicative of future results.
                * This model serves as a demonstration and should not be used for financial advice.
                """)

