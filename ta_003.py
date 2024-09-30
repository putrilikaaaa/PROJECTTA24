import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Fungsi untuk mengupload file
def upload_file():
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    else:
        return None

# Fungsi untuk memproses dan memvisualisasikan data
def process_data(data_df):
    try:
        # Mengubah kolom 'Tanggal' menjadi format datetime
        data_df['Tanggal'] = pd.to_datetime(data_df['Tanggal'], format='%d-%b-%y')

        # Mengatur kolom 'Tanggal' sebagai index
        data_df.set_index('Tanggal', inplace=True)

        # Menghapus kolom non-numerik jika ada
        data_df = data_df.select_dtypes(include=[float, int])

        return data_df
    except Exception as e:
        st.error(f"Error dalam memproses data: {e}")
        return None

# Fungsi untuk menampilkan statistik deskriptif
def show_descriptive_statistics(data_df):
    st.subheader("Statistika Deskriptif")
    st.write(data_df.describe())

# Fungsi untuk plot time series harian
def plot_time_series_daily(data_df: pd.DataFrame, province: str):
    st.subheader(f"Plot Time Series Harian untuk {province}")
    if province in data_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(data_df.index, data_df[province], label=province, color='blue')
        plt.title(f"Time Series Harian - {province}", fontsize=16)
        plt.xlabel('Tanggal', fontsize=12)
        plt.ylabel('Nilai', fontsize=12)
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot()

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Statistika Deskriptif dan Plot Time Series")

    # Upload file dan proses data
    data_df = upload_file()

    if data_df is not None:
        # Menampilkan data
        st.subheader("Dataframe")
        st.write(data_df)

        # Memproses data
        processed_data_df = process_data(data_df)

        if processed_data_df is not None:
            # Menambahkan dropdown untuk memilih provinsi
            selected_province = st.selectbox("Pilih Provinsi", options=processed_data_df.columns.tolist())

            # Menampilkan statistik deskriptif berdasarkan provinsi yang dipilih
            if selected_province:
                st.subheader(f"Statistika Deskriptif untuk {selected_province}")
                st.write(processed_data_df[selected_province].describe())

                # Menampilkan plot time series harian untuk provinsi yang dipilih
                plot_time_series_daily(processed_data_df, selected_province)

if __name__ == "__main__":
    main()
