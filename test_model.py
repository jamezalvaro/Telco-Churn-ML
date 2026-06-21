import joblib
import pandas as pd

def main():
    print("Memuat model...")
    try:
        model_data = joblib.load('churn_model.joblib')
        model = model_data['model']
        train_median = model_data['train_median_total_charges']
        print("Model berhasil dimuat!")
    except FileNotFoundError:
        print("Error: 'churn_model.joblib' tidak ditemukan. Pastikan Anda telah menjalankan train_and_save_model.py terlebih dahulu.")
        return

    print("\nMenyiapkan data uji coba sintetik...")
    # Membuat 2 contoh data uji (satu contoh kemungkinan Churn tinggi, satu rendah)
    sample_data = pd.DataFrame({
        'Contract': ['Month-to-month', 'Two year'],
        'InternetService': ['Fiber optic', 'No'],
        'TotalCharges': ['100.50', ' '], # Mensimulasikan satu nilai kosong
        'tenure': [2, 60],
        'PaperlessBilling': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No'],
        'StreamingMovies': ['No', 'No internet service']
    })

    print("Data Uji:")
    print(sample_data)

    # Preprocessing untuk memastikan tidak ada nilai kosong pada TotalCharges
    sample_data['TotalCharges'] = pd.to_numeric(sample_data['TotalCharges'], errors='coerce')
    sample_data['TotalCharges'] = sample_data['TotalCharges'].fillna(train_median)

    print("\nMelakukan Prediksi...")
    predictions = model.predict(sample_data)
    prediction_probs = model.predict_proba(sample_data)[:, 1]

    for i in range(len(predictions)):
        churn_status = "Yes" if predictions[i] == 1 else "No"
        print(f"Data {i+1} -> Prediksi Churn: {churn_status} (Probabilitas Churn: {prediction_probs[i]*100:.2f}%)")

if __name__ == '__main__':
    main()
