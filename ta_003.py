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

        # Mengubah data menjadi bulanan dengan rata-rata
        data_monthly = data_df.resample('M').mean()

        # Memfilter data untuk hanya mengambil bulan April sampai November
        data_monthly_filtered = data_monthly[data_monthly.index.month.isin(range(4, 12))]

        return data_monthly_filtered
    except Exception as e:
        st.error(f"Error dalam memproses data: {e}")
        return None

# Fungsi untuk menampilkan statistik deskriptif
def show_descriptive_statistics(data_df):
    st.subheader("Statistika Deskriptif")
    st.write(data_df.describe())

# Fungsi untuk plot time series bulanan
def plot_time_series_monthly(data_df: pd.DataFrame, interval: int = 3):
    st.subheader("Plot Time Series Bulanan")
    if data_df is not None:
        num_provinces = data_df.shape[1]  # Jumlah kolom provinsi (semua kolom numerik)
        cols = 3  # Jumlah kolom untuk subplot
        rows = -(-num_provinces // cols)  # Menghitung jumlah baris

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))

        for i, (column, ax) in enumerate(zip(data_df.columns, axes.flatten())):
            ax.plot(data_df.index, data_df[column], label=column, color='blue')
            ax.set_title(column, fontsize=12)
            ax.set_xlabel('Tanggal', fontsize=10)
            ax.set_ylabel('Nilai', fontsize=10)
            ax.legend(loc='upper left')

            # Menampilkan label x hanya pada interval tertentu
            xticks = data_df.index[::interval]
            ax.set_xticks(xticks)  # Mengatur posisi label
            ax.set_xticklabels(xticks.strftime('%b %Y'), rotation=45)  # Mengatur format tanggal

        # Menghapus sumbu yang tidak digunakan jika jumlah provinsi tidak memenuhi semua kotak
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()
        st.pyplot(fig)

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
        data_monthly_filtered = process_data(data_df)

        if data_monthly_filtered is not None:
            # Menampilkan statistik deskriptif
            show_descriptive_statistics(data_monthly_filtered)

            # Menampilkan plot time series
            plot_time_series_monthly(data_monthly_filtered, interval=3)

if __name__ == "__main__":
    main()
