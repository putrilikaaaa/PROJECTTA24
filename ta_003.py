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
    statistics = data_df.describe().transpose()  # Transpose untuk mengubah orientasi menjadi baris

    # Membuat kolom untuk menampilkan statistik deskriptif
    cols = st.columns(len(statistics))
    
    for col, (stat_name, stat_values) in zip(cols, statistics.iterrows()):
        col.subheader(stat_name)
        col.write(f"Count: {stat_values['count']}")
        col.write(f"Mean: {stat_values['mean']:.2f}")
        col.write(f"Std: {stat_values['std']:.2f}")
        col.write(f"Min: {stat_values['min']:.2f}")
        col.write(f"25%: {stat_values['25%']:.2f}")
        col.write(f"50%: {stat_values['50%']:.2f}")
        col.write(f"75%: {stat_values['75%']:.2f}")
        col.write(f"Max: {stat_values['max']:.2f}")

# Fungsi untuk plot time series harian
def plot_time_series_daily(data_df: pd.DataFrame, province: str):
    st.subheader(f"Plot Time Series Harian untuk {province}")
    if province in data_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))  # Buat objek figure dan axis
        ax.plot(data_df.index, data_df[province], label=province, color='blue')
        ax.set_title(f"Time Series Harian - {province}", fontsize=16)
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Nilai', fontsize=12)
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)  # Kirimkan objek figure ke st.pyplot

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

        if processed_data_df is not None and not processed_data_df.empty:
            # Menambahkan dropdown untuk memilih provinsi
            selected_province = st.selectbox("Pilih Provinsi", options=processed_data_df.columns.tolist())

            # Menampilkan statistik deskriptif berdasarkan provinsi yang dipilih
            if selected_province:
                st.subheader(f"Statistika Deskriptif untuk {selected_province}")
                show_descriptive_statistics(processed_data_df[[selected_province]])

                # Menampilkan plot time series harian untuk provinsi yang dipilih
                plot_time_series_daily(processed_data_df, selected_province)

if __name__ == "__main__":
    main()
