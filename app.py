import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def process_stock_data(file_path, stock_name):
    """Process individual stock dataset and return model metrics"""
    try:
        # Load dataset
        data = pd.read_csv(r"C:\Users\krush\Desktop\Internship\FUTURE_ML_02\Dataset\symbols_valid_meta.csv", encoding="utf-8", on_bad_lines="skip")
        
        # Verify required columns exist
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Required: {required_columns}")
        
        # Clean and convert Date column
        try:
            # First try parsing dates as is
            data['Date'] = pd.to_datetime(data['Date'])
        except:
            try:
                # Try parsing with different date formats
                data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
            except:
                print(f"Error: Invalid date format in {stock_name}")
                return None
        
        # Drop any rows with invalid dates
        data = data.dropna(subset=['Date'])
        
        if len(data) == 0:
            raise ValueError(f"No valid data remaining after cleaning for {stock_name}")
            
        # Convert dates to ordinal numbers
        data['Date'] = data['Date'].map(pd.Timestamp.toordinal)
        
        # Feature selection
        X = data[['Date', 'Open', 'High', 'Low', 'Volume']]
        y = data['Close']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual Prices', color='blue')
        plt.plot(y_pred, label='Predicted Prices', color='red')
        plt.legend()
        plt.title(f'Stock Price Prediction - {stock_name}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.savefig(f'prediction_{stock_name}.png')
        plt.close()
        
        return {
            'stock_name': stock_name,
            'mse': mse,
            'r2_score': r2,
            'model': model
        }
    
    except Exception as e:
        print(f"Error processing {stock_name}: {str(e)}")
        return None

def main():
    # Directory containing your dataset files
    data_dir = "datasets"  # Create this folder and put your CSV files inside
    results = []
    
    # Process each dataset
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            stock_name = file.replace('.csv', '')
            result = process_stock_data(file_path, stock_name)
            if result:
                results.append(result)
    
    # Display summary of results
    print("\nSummary of Results:")
    print("-" * 50)
    for result in results:
        print(f"\nStock: {result['stock_name']}")
        print(f"Mean Squared Error: {result['mse']:.2f}")
        print(f"R² Score: {result['r2_score']:.4f}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    names = [r['stock_name'] for r in results]
    mse_values = [r['mse'] for r in results]
    
    plt.bar(names, mse_values)
    plt.title('Model Performance Comparison')
    plt.xlabel('Stocks')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.show()

if __name__ == "__main__":
    main()