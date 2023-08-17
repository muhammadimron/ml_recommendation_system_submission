# Laporan Proyek Machine Learning - Muhammad Imron
#
## Project Overview
Perkembangan pesat teknologi internet telah menghasilkan pertumbuhan eksplosif dari sistem informasi. Salah satu dari hal tersebut adalah sistem rekomendasi. Tujuan dari sistem rekomendasi adalah untuk secara otomatis menghasilkan item yang disarankan (film, buku, berita, musik, CD, DVD, halaman web) untuk pengguna menurut preferensi historis mereka. Kehadiran sistem rekomendasi membawa profit besar. Hal tersebut dapat dilihat dari berbagai macam platform aplikasi seperti NetFLix, Spotify, dan Amazon yang menjadikan sistem rekomendasi sebagai salah satu fitur layanan mereka.

Banyak penelitian telah dilakukan dalam mengembangkan sistem rekomendasi yang efektif dan tepat sasaran. Salah satu terobosan ilmu pengetahuan yang menjadi fundamental sistem rekomendasi adalah diperkenalkannya teknik *content based filtering* oleh Michael *et al* [1] dan teknik *collaborative filtering* oleh Sarwar *et al* [2]. Tujuan utama dari proyek ini adalah membuat sistem rekomendasi film menggunakan dataset dari website kaggle yang dibuat oleh seorang pengguna dengan *username* shinigami [3]. Dataset terebut berisi mengenai informasi film dan daftar penilaian film oleh pengguna.

## Business Understandings
### *Problem Statements*
Berdasarkan *project overview* yang telah diuraikan sebelumnya, proyek ini akan mengembangkan sebuah sistem rekomendasi film dengan menjawab permasalahan berikut.
- Bagaimana menyediakan daftar film yang disarankan tanpa preferensi apapun?
- Bagaimana menyediakan daftar film yang disarankan menggunakan preferensi fitur dari film?
- Bagaimana menyediakan daftar film yang disarankan menggunakan preferensi historis pengguna?

### *Goals*
Untuk  menjawab pertanyaan tersebut, proyek ini akan membuat sebuah sistem rekomendasi film dengan tujuan atau goals sebagai berikut:
- Membuat sejumlah daftar rekomendasi film menggunakan pendekatan *non-personalized recommendation*.
- Menghasilkan sejumlah rekomendasi film yang dipersonalisasi untuk pengguna dengan teknik *content-based filtering*.
- Menghasilkan sejumlah rekomendasi restoran yang sesuai dengan preferensi di masa lalu dengan teknik *collaborative filtering*.

### *Solution Statements*
Untuk  meraih tujuan tersebut, proyek ini akan mengimplementasikan hal berikut:
- Mengimplementasikan *Exploratory Data Analysis* (EDA) untuk mengetahui fitur-fitur yang dapat digunakan sebagai preferensi untuk sistem rekomendasi.
- Membuat algoritma fungsi untuk menghasilkan *non-personalized recommendation*.
- Membuat algoritma fungsi untuk menghasilkan sistem rekomendasi *content-based filtering* dengan *library CountVectorizer* dan *Cosine Similarity*.
- Membuat algoritma *class* yang diwariskan dari model Keras API untuk menghasilkan sistem rekomendasi *collaborative filtering*.
- Menggunakan matriks *Normalize Discount Cumulative Gain* dan *Root Mean Squared Error* sebagai matriks evaluasi untuk sistem rekomendasi *content-based filtering* dan *collaborative filtering*.

## Data Understandings
Dataset yang digunakan dalam proyek ini adalah dataset yang diambil dari website [kaggle]. Dataset tersebut berisi data mengenai detail informasi film dan informasi penilaian film yang diberikan oleh pengguna. Dataset tersebut memiliki dua variabel yang berbentuk file berformat csv, yaitu movies dan ratings. Penjelasan lengkap mengenai variabel dalam dataset dapat dilihat pada list di bawah.

**Variabel-variabel pada _Movie Recommender System Dataset_ adalah sebagai berikut:**
*   movies: merupakan data yang berisi mengenai detail informasi film
*   ratings: merupakan data yang berisi mengenai informasi penilaian film oleh pengguna

### *Exploratory Data Analysis* (EDA)
Pada proyek ini, terdapat 3 langkah EDA yang dilakukan. Ketiga langkah tersebut adalah sebagai berikut:
1. **Deskripsi variabel**
Pada tahap ini, variabel-variabel dalam *Movie Recommender System Dataset* dijabarkan menggunakan fungsi *info()*. Penjabaran ini dilakukan untuk memahami kondisi awal variabel-variabel dalam *Movie Recommender System Dataset*. Berikut adalah penjabaran variabel *movies*.

    Tabel 1. Informasi awal *movies*.
    | # | Column  | Non-Null Count | Dtype  |
    |:-:|---------|----------------|--------|
    | 0 | movieId | 9742 non-null  | int64  |
    | 1 | title   | 9742 non-null  | object |
    | 2 | genres  | 9742 non-null  | object |
    
    Berdasarkan tabel 1 di atas, dapat diketahui bahwa variabel *movies* memiliki 9742 entri. Fitur-fitur dalam variabel *movies* adalah sebagai berikut:
    - movieId: id movie
    - title: judul movie
    - genres: daftar genre movie
    
    Berdasarkan tabel 1 juga dapat diketahui bahwa variabel *movies* memiliki dua tipe data, yaitu *int64* pada fitur *movieId* dan *object* pada fitur *title* dan *genres*. Setelah penjabaran variabel *movies*, selanjutnya adalah penjabaran variabel *ratings*.
    
    Tabel 2. Informasi awal *ratings*.
    | # | Column    | Non-Null Count  | Dtype   |
    |:-:|-----------|-----------------|---------|
    | 0 | userId    | 100836 non-null | int64   |
    | 1 | movieId   | 100836 non-null | int64   |
    | 2 | rating    | 100836 non-null | float64 |
    | 3 | timestamp | 100836 non-null | int64   |
    
    Berdasarkan tabel 2 di atas, dapat diketahui bahwa variabel *ratings* memiliki 100836 entri. Fitur-fitur dalam variabel *ratings* adalah sebagai berikut:
    - userId: id pengguna
    - movieId: id movie
    - rating: jumlah penilain movie terhadap pengguna
    - timestamp: waktu ketika entri diciptakan yang berbentuk detik sejak 1 Januari 1970
    
    Berdasarkan tabel 2 juga dapat diketahui bahwa variabel *ratings* memiliki dua tipe data, yaitu *float64* pada fitur *rating* dan *int64* pada fitur *userId*, *movieId* dan *timestamp*.
    
    Pada tahap ini, telah dilakukan deskripsi variabel *movies* dan *ratings*. Selanjutnya adalah tahap pembersihan data atau *data cleaning* pada variabel *movies* dan *ratings*.
    
2. **_Data cleaning_**
Pada tahap ini akan dilakukan penggabungan variabel antara *movies* dan *ratings* menjadi satu dataset. Setelah dilakukan penggabungan antara *movies* dan *ratings* menjadi satu dataset, akan dilakukan 'pembersihan' data yang kotor sehingga terbentuk satu dataset bersih untuk sistem rekomendasi yang akan dibangun dalam proyek ini.

    Langkah pertama yang dilakukan pada tahap *data cleaning* adalah penggabungan variabel antara *movies* dan *ratings* menjadi satu dataset. Penggabungan tersebut dilakukan dengan memanfaatkan sebuah fungsi bernama *merge* yang berasal dari *library pandas*. Penggabungan variabel menggunakan kolom *movieId* sebagai acuan kolom ketika variabel digabungkan. Hasil penggabungan dapat dilihat pada tabel 3.
    
    Tabel 3. Hasil penggabungan variabel
    | # | userId | movieId | rating | timestamp  | title            | genres                                          |
    |:-:|--------|---------|--------|------------|------------------|-------------------------------------------------|
    | 0 | 1      | 1       | 4.0    | 964982703  | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
    | 1 | 5      | 1       | 4.0    | 847434962  | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
    | 2 | 7      | 1       | 4.5    | 1106635946 | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
    | 3 | 15     | 1       | 2.5    | 1510577970 | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
    | 4 | 17     | 1       | 4.5    | 1305696483 | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
    
    Berdasarkan tabel 3, dapat dilihat bahwa variabel *movies* dan *ratings* telah berhasil digabung. Untuk memudahkan penjelasan, mari sebut saja dataset hasil penggabungan variabel *movies* dan *ratings* adalah dataset *df*.
    
    Setelah membuat dataset *df*, langkah selanjutnya adalah memeriksa informasi terkini dari dataset *df*. Pemeriksaan informasi terkini dilakukan dengan memanggil fungsi *info* dan *describe*. Hasil pemanggilan fungsi *info* dapat dilihat pada tabel 4.
    
    Tabel 4. Informasi awal dataset *df*
    | # | Column    | Non Null Count  | Dtype   |
    |:-:|-----------|-----------------|---------|
    | 0 | userId    | 100836 non-null | int64   |
    | 1 | movieId   | 100836 non-null | int64   |
    | 2 | rating    | 100836 non-null | float64 |
    | 3 | timestamp | 100836 non-null | int64   |
    | 4 | title     | 100836 non-null | object  |
    | 5 | genres    | 100836 non-null | object  |
    
    Berdasarkan tabel 4, dapat dilihat bahwa dataset *df* memiliki enam kolom yang berasal dari variabel *movies* dan *ratings*. Kolom-kolom tersebut adalah sebagai berikut.
    - userId: id pengguna
    - movieId: id movie
    - rating: jumlah penilain movie terhadap pengguna
    - timestamp: waktu ketika entri diciptakan yang berbentuk detik sejak 1 Januari 1970
    - title: judul movie
    - genres: daftar genre movie
    
    Berdasarkan tabel 4 juga, dapat diambil informasi bahwa dataset *df* memiliki 100836 entrI dan tiga tipe data, yaitu *int64* pada kolom *userId*, *movieId*, dan *timestamp*, kemudian tipe data *float64* pada kolom *rating*, dan tipe data *object* pada kolom *title* dan *genres*.
    
    Setelah melakukan pemeriksaan informasi menggunakan fungsi *info*, selanjutnya adalah melakukan pemeriksaan statistik menggunakan fungsi *describe*. Hasil pemeriksaan statistik dapat dilihat pada tabel 5.
    
    Tabel 5. Informasi statistik dataset *df*
    |   #   | userId        | movieId       | rating        | timestamp    |
    |:-----:|---------------|---------------|---------------|--------------|
    | count | 100836.000000 | 100836.000000 | 100836.000000 | 1.008360e+05 |
    | mean  | 326.127564    | 19435.295718  | 3.501557      | 1.205946e+09 |
    | std   | 182.618491    | 35530.987199  | 1.042529      | 2.162610e+08 |
    | min   | 1.000000      | 1.000000      | 0.500000      | 8.281246e+08 |
    | 25%   | 177.000000    | 1199.000000   | 3.000000      | 1.019124e+09 |
    | 50%   | 325.000000    | 2991.000000   | 3.500000      | 1.186087e+09 |
    | 75%   | 477.000000    | 8122.000000   | 4.000000      | 1.435994e+09 |
    | max   | 610.000000    | 193609.000000 | 5.000000      | 1.537799e+09 |
    
    Berdasarkan tabel 5, dapat terlihat informasi statistik dari kolom *userId*, *movieId*, *rating*, dan *timestamp*. Beberapa informasi yang dapat diambil adalah sebagai berikut.
    - Kolom *userId* memiliki id pengguna terkecil 1 dan id pengguna terbesar 610
    - Kolom *movieId* memiliki id film terkecil 1 dan id film terbesar 193609
    - Kolom *rating* memiliki kategori penilaian terkecil 0.5 dan kategori penilaian terbesar 5
    - Kolom *timestamp* memiliki nilai terkecil 828124600 detik dan nilai terbesar 1537799000 detik
    
    Berdasarkan representasi informasi yang diberikan baik pada tabel 4 dan tabel 5, sepertinya tidak ada suatu keanehan. Untuk berjaga-jaga, maka jumlah nilai *null* pada setiap kolom juga perlu diperiksa. Pemeriksaan *missing value* memanfaatkan fungsi *isnull* dan fungsi *sum*. Hasil pemeriksaan dapat dilihat pada tabel 6.
    
    Tabel 6. Jumlah nilai *null* pada dataset *df*
    | # | Column    | Null Count |
    |:-:|-----------|------------|
    | 0 | userId    | 0          |
    | 1 | movieId   | 0          |
    | 2 | rating    | 0          |
    | 3 | timestamp | 0          |
    | 4 | title     | 0          |
    | 5 | genres    | 0          |
    
    Berdasarkan tabel 6, dapat terlihat bahwa tidak ada nilai *null* pada seluruh kolom dataset *df*. Informasi pada tabel 4, 5, dan 6 menunjukkan bahwa tidak ada keanehan data seperti *missing values* dan jumlah minimal pada kolom numerik yang tidak normal.
    
    Langkah selanjutnya adalah memeriksa format nilai pada dataset *df*. Berdasarkan tabel 4 dan 5, format nilai yang ada pada kolom numerik sudah sesuai dengan apa yang diinginkan. Berdasarkan tabel 3, format nilai pada kolom *genres* juga sudah sesuai dengan apa yang diinginkan. Namun tidak dengan format nilai pada kolom *title*. Pertama, dilakukan pemeriksaan terlebih dahulu pada kolom *title*. Hasil pemeriksaan kolom *title* dapat dilihat pada tabel 7.
    
    Tabel 7. Kolom *title* pada dataset *df*
    |    #   | title                            |
    |:------:|----------------------------------|
    | 0      | Toy Story (1995)                 |
    | 1      | Toy Story (1995)                 |
    | 2      | Toy Story (1995)                 |
    | 3      | Toy Story (1995)                 |
    | 4      | Toy Story (1995)                 |
    | ...    | ...                              |
    | 100831 | Bloodmoon (1997)                 |
    | 100832 | Sympathy for the Underdog (1971) |
    | 100833 | Hazard (2005)                    |
    | 100834 | Blair Witch (2016)               |
    | 100835 | 31 (2016)                        |
    
    Berdasarkan tabel 7, maka dapat diambil informasi bahwa kolom *title* masih mengandung nilai tahun. Sebaiknya kedua nilai tersebut dipisah sehingga tidak ada penggabungan nilai judul dan tahun. Untuk memisahkan kedua nilai tersebut, dilakukan pemanggilan fungsi *str.extract* untuk mengekstraksi nilai tahun pada kolom *title*, dan dilakukan pemanggilan fungsi *str.split* untuk menghapus nilai tahun pada kolom *title*. Hasil pemisahan nilai tahun pada kolom *title* dapat dilihat pada tabel 8.
    
    Tabel 8. Hasil pemisahan nilai tahun dan judul
    | # | userId | movieId | rating | timestamp  | title     | genres                                          | year |
    |:-:|--------|---------|--------|------------|-----------|-------------------------------------------------|------|
    | 0 | 1      | 1       | 4.0    | 964982703  | Toy Story | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1995 |
    | 1 | 5      | 1       | 4.0    | 847434962  | Toy Story | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1995 |
    | 2 | 7      | 1       | 4.5    | 1106635946 | Toy Story | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1995 |
    | 3 | 15     | 1       | 2.5    | 1510577970 | Toy Story | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1995 |
    | 4 | 17     | 1       | 4.5    | 1305696483 | Toy Story | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1995 |

    Berdasarkan tabel 8, dapat terlihat bahwa kolom *title* sekarang tidak memiliki nilai tahun lagi. Nilai tahun sekarang berada di kolom baru, yaitu kolom *year*.
    
    Perubahan pada struktur dataset *df* juga akan membuat informasi dalam dataset *df* berubah. Pertama, mari cek terlebih dahulu informasi pada dataset *df* menggunakan fungsi *info*. Hasil pengecekan informasi pada dataset *df* dapat dilihat pada tabel 9.
    
    Tabel 9. Informasi dataset *df* setelah pemisahan nilai tahun
    | # | Column    | Non-Null Count  | Dtype   |
    |:-:|-----------|-----------------|---------|
    | 0 | userId    | 100836 non-null | int64   |
    | 1 | movieId   | 100836 non-null | int64   |
    | 2 | rating    | 100836 non-null | float64 |
    | 3 | timestamp | 100836 non-null | int64   |
    | 4 | title     | 100836 non-null | object  |
    | 5 | genres    | 100836 non-null | object  |
    | 6 | year      | 100819 non-null | object  |
    
    Berdasarkan tabel 9, maka dapat diambil informasi bahwa sekarang kolom pada dataset *df* telah berubah menjadi tujuh kolom. Kolom baru merupakan kolom *year* yang memiliki nilai tahun produksi film. Berdasarkan tabel 9 juga, dapat diambil informasi bahwa kolom *year* memiliki nilai entri 100819 dan bertipe data *object*. Hal tersebut menunjukkan sebuah abnormalitas. Entri 100819 lebih sedikit dibanding entri kolom lain sehingga dapat diasumsikan terdapat *missing value*. Kemudian tipe data kolom *year* yang bertipe *object* juga seharusnya tidak terjadi karena kolom *year* hanyalah sebuah angka. Mari periksa nilai unik pada kolom *year*. Pemeriksaan nilai unik pada kolom *year* dilakukan dengan cara memanggil fungsi *unique*. Hasil pemeriksaan nilai unik pada kolom *year* dapat dilihat pada gambar 1.
    
    ![year_unclean.jpg](https://drive.google.com/uc?export=view&id=1rqHyVvGtpQeuRSPVtmh50NvZ33QkI07r "year_unclean")
    Gambar 1. Nilai unik pada kolom *year*
    
    Berdasarkan gambar 1 di atas, maka dapat terlihat terdapat abnormalitas pada kolom *year* yang tidak sesuai. Abnormalitas tersebut adalah sebagai berikut.
    - Terdapat nilai *nan* dan nilai '2006-2007'
    - Nilai pada kolom *year* berbentuk *string*
    
    Untuk mengatasi hal tersebut, maka dilakukan hal berikut.
    - Memanggil fungsi *dropna* untuk menghapus nilai *nan* pada kolom *year*
    - Memanggil fungsi *drop* untuk menghapus nilai '2006-2007' pada kolom *year*
    - Mengubah tipe data kolom *year* menjadi *int64* dengan fungsi *astype*
    
    Setelah melakukan hal di atas, seharusnya kolom *year* sudah 'bersih'. Untuk mengecek hal tersebut, mari periksa kembali menggunakan fungsi *info*. Hasil pemeriksaan menggunakan fungsi *info* dapat dilihat pada tabel 10.
    
    Tabel 10. Informasi dataset *df* setelah *data cleaning*
    | # | Column    | Non-Null Count  | Dtype   |
    |:-:|-----------|-----------------|---------|
    | 0 | userId    | 100818 non-null | int64   |
    | 1 | movieId   | 100818 non-null | int64   |
    | 2 | rating    | 100818 non-null | float64 |
    | 3 | timestamp | 100818 non-null | int64   |
    | 4 | title     | 100818 non-null | object  |
    | 5 | genres    | 100818 non-null | object  |
    | 6 | year      | 100818 non-null | int64   |
    
    Berdasarkan tabel 10, maka dapat diambil kesimpulan bahwa sekarang dataset *df* sekarang sudah 'bersih'. Hal tersebut dapat terlihat dari jumlah entri yang sama dan tipe data yang sesuai pada setiap kolom. Langkah selanjutnya setelah melakukan *data cleaning* adalah analisis fitur.
    
3. **Analisis Fitur**
Pada tahap ini, akan dilakukan analisis fitur dataset *df* yang berkemungkinan menjadi preferensi untuk sistem rekomendasi. Terdapat dua fitur yang berpotensial untuk menjadi preferensi sistem rekomendasi yang akan dibuat pada proyek ini, yaitu *rating* dan *genres*. Pertama, mari lakukan analisis terlebih dahulu fitur *rating* pada dataset *df*.

    Terdapat dua hal pada fitur *rating* yang akan dianalisis. Hal pertama adalah jumlah frekuensi fitur *rating*. Visualisasi frekuensi fitur *rating* dapat dilihat pada gambar 2.
    
    ![rating_frequent.jpg](https://drive.google.com/uc?export=view&id=1qzEyaLBgX3IJ4n0bfq7Jx6Pnw4B6Gf_s "rating_frequent")
    Gambar 2. Visualisasi frekuensi *rating*
    
    Berdasarkan gambar 2, maka dapat diambil dua informasi. Informasi pertama adalah nilai *rating* terendah adalah 0.5 dan nilai *rating* tertinggi adalah 5 dan informasi kedua adalah frekuensi rating terbanyak adalah 4 dan frekuensi rating terendah adalah 0.5
    
    Hal kedua yang dianalisis pada fitur *rating* adalah distribusi frekuensi *rating* pada setiap film. Visualisasi distribuse frekuensi *rating* pada setiap film dapat dilihat pada gambar 3.
    
    ![rating_distribution.jpg](https://drive.google.com/uc?export=view&id=1Q26ID_aTlF1U0Xy8mBk1_2PllTa-VKgI "rating_distribution")
    Gambar 3. Distribusi frekuensi *rating* pada setiap film
    
    Berdasarkan gambar 3, maka terdapat beberapa informasi yang dapat diambil. Informasi pertama adalah 3000 merupakan angka terbanyak jumlah rating yang dimiliki oleh suatu film. Informasi lainnya, terdapat film yang dinilai kurang dari 10 kali. Film-film tersebut kurang berpengaruh terhadap sistem rekomendasi yang akan dibuat sehingga film yang dinilai kurang dari 10 kali akan dihapus. Penghapusan film tersebut dilakukan dengan memanfaatkan fungsi *isin* yang menggunakan parameter film yang dinilai lebih dari 10 kali. Film dengan jumlah *rating* terendah setelah melakukan penghapusan dapat dilihat pada tabel 11.
    
    Tabel 11. Distribusi *rating* terendah terhadap film
    | # | Title             | Rating Frequency |
    |:-:|-------------------|------------------|
    | 0 | Skulls, The       | 10               |
    | 1 | Doom              | 10               |
    | 2 | Urban Legend      | 10               |
    | 3 | Detroit Rock City | 10               |
    | 4 | Fast Five         | 10               |
    
    Berdasarkan tabel 11, maka dapat disimpulkan bahwa sekarang distribusi frekuensi *rating* terendah berjumlah 10 kali. Hal tersebut menunjukkan bahwa penghapusan distribusi frekuensi *rating* dibawah 10 kali telah berhasil dilakukan.
    
    Setelah melakukan analisis terhadap fitur *rating*, selanjutnya adalah melakukan analisis terhadap fitur *genres*. Hal yang dianalisis pada fitur *genres* adalah frekuensi fitur *genres* pada dataset *df*. Hasil visualisasi frekuensi fitur *genres* pada dataset *df* dapat dilihat pada gambar 4.
    
    ![genres_frequent.jpg](https://drive.google.com/uc?export=view&id=13HEXMVeu4zLOu6rrzRbFmpq4SaHsxcKb "genres_frequent")
    Gambar 4. Frekuensi *genres* dalam dataset *df*
    
    Berdasarkan gambar 4, maka dapat diambil beberapa informasi. Informasi yang dapat diambil adalah frekuensi *genres* terbanyak adalah film yang memiliki *genre* drama, sedangkan film yang memiliki *genre* terendah adalah film yang memiliki *genre documentary*.
    
    Akhirnya tahap analisis fitur telah selesai sekaligus menandai bahwa tahap EDA juga telah selesai. Sebelum melangkah ke tahap selanjutnya yaitu *data preparation*, mari cek kembali informasi akhir dataset *df*. Seperti biasa, pengeceken informasi dilakukan dengan memanggil fungsi *info* dan *describe*. Hasil pengecekan informasi menggunakan fungsi *info* dapat dilihat pada tabel 12.
    
    Tabel 12. Informasi akhir dataset *df*
    | # | Column    | Non-Null Count | Dtype   |
    |:-:|-----------|----------------|---------|
    | 0 | userId    | 81116 non-null | int64   |
    | 1 | movieId   | 81116 non-null | int64   |
    | 2 | rating    | 81116 non-null | float64 |
    | 3 | timestamp | 81116 non-null | int64   |
    | 4 | title     | 81116 non-null | object  |
    | 5 | genres    | 81116 non-null | object  |
    | 6 | year      | 81116 non-null | int64   |
    
    Hasil pengecekan informasi statistik kolom numerik pada dataset *df* dapat dilihat pada tabel 13.
    
    Tabel 13. Informasi statistik akhir kolom numerik dataset *df*
    |   #   | userId       | movieId       | rating       | timestamp    | year         |
    |:-----:|--------------|---------------|--------------|--------------|--------------|
    | count | 81116.000000 | 81116.000000  | 81116.000000 | 8.111600e+04 | 81116.000000 |
    | mean  | 318.989977   | 14857.178078  | 3.573678     | 1.197217e+09 | 1994.385571  |
    | std   | 181.748877   | 29539.336412  | 1.018590     | 2.167182e+08 | 13.233176    |
    | min   | 1.000000     | 1.000000      | 0.500000     | 8.281246e+08 | 1922.000000  |
    | 25%   | 167.000000   | 1007.000000   | 3.000000     | 1.001562e+09 | 1990.000000  |
    | 50%   | 316.000000   | 2471.000000   | 4.000000     | 1.180447e+09 | 1996.000000  |
    | 75%   | 474.000000   | 6016.000000   | 4.000000     | 1.431955e+09 | 2002.000000  |
    | max   | 610.000000   | 187593.000000 | 5.000000     | 1.537799e+09 | 2018.000000  |
    
    Berdasarkan informasi pada tabel 12 dan 13, maka dapat disimpulkan bahwa dataset *df* setelah melalui EDA memiliki 81116 entri dan 7 kolom. Terdapat tiga tipe data pada dataset *df*, yaitu *int64* untuk kolom *userId*, *movieId*, *timestamp*, dan *year*, kemudian tipe data *float64* untuk kolom *rating*, dan tipe data *object* untuk kolom *title* dan *genres*. Tahap selanjutnya adalah *data preparation*.
    
## Data Preparation
Pada proyek ini, terdapat tiga sistem rekomendasi yang akan dibangun. Ketiga sistem rekomendasi tersebut adalah sebagai berikut.
- Sistem rekomendasi yang tidak dipersonalisasi yang selanjutnya mari sebut dengan istilah *non-personalized recommendation*
- Sistem rekomendasi yang dipersonalisasi dengan teknik *content-based filtering*. Selanjutnya mari sebut dengan istilah *content-based recommendation*
- Sistem rekomendasi yang dipersonalisasi dengan teknik *collaborative filtering*. Selanjutnya mari sebut dengan istilah *collaborative recommendation*

Masing-masing dari ketiga sistem rekomendasi di atas membutuhkan kombinasi dataset dengan bentuk yang berbeda. Fokus dari tahap ini adalah membangun dataset untuk ketiga sistem rekomendasi di atas.

### *Preprocessing*
*Preprocessing* merupakan sebuah teknik yang dilakukan pada *data preparation*. *Preprocessing* ini mengubah formasi dan bentuk dari dataset yang telah melewati tahap *data cleaning* menjadi dataset yang sesuai dengan input model. *Preprocessing* ini perlu dilakukan untuk membuat dataset yang sesuai dengan masing-masing sistem rekomendasi yang akan dibuat pada proyek ini. Pertama, mari terapkan *preprocessing* untuk *non-personalized recommendation*.
1. *Non-personalized recommendation*
*Non-personalized recommendation* merupakan sistem rekomendasi yang tidak membutuhkan fitur spesifik. Sistem rekomendasi ini biasanya digunakan pada *homepage* aplikasi. Sebagai contoh adalah produk terpopuler pada *homepage* aplikasi *e-commerce*. 

    Dataset yang dibutuhkan pada sistem rekomendasi ini adalah dataset yang memiliki sorting descending dalam fitur populer. Untuk membuat dataset sesuai dengan kondisi tersebut, maka fitur yang dibutuhkan hanyalah fitur *title* dan *rating* saja sehingga pertama-tama, mari ambil data pada kolom *title* dan *rating* dan jadikan kumpulan data tersebut menjadi dataset baru yang bernama *non_personalized_df*. Hasil pembuatan dataset *non_personalized_df* dapat dilihat pada tabel 14.
    
    Tabel 14. Dataset *non_personalized_df* (awal)
    | # | title     | rating |
    |:-:|-----------|--------|
    | 0 | Toy Story | 4.0    |
    | 1 | Toy Story | 4.0    |
    | 2 | Toy Story | 4.5    |
    | 3 | Toy Story | 2.5    |
    | 4 | Toy Story | 4.5    |
    
    Berdasarkan tabel 14, masih terdapat nilai duplikat sehingga diperlukan penghapusan nilai duplikat menggunakan fungsi *drop_duplicate*. Hasil dataset *non_personalized_df* setelah penghapusan nilai duplikat dapat dilihat pada tabel 15.
    
    Tabel 15. Dataset *non_personalized_df* (akhir)
    | # | title     | rating |
    |:-:|-----------|--------|
    | 0 | Toy Story | 4.0    |
    | 1 | Toy Story | 4.5    |
    | 2 | Toy Story | 2.5    |
    | 3 | Toy Story | 3.5    |
    | 4 | Toy Story | 3.0    |
    
    Berdasarkan tabel 15, sudah tidak ada nilai duplikat lagi. Selanjutnya adalah membuat dataset yang dibutuhkan untuk *content-based recommendation*.

2. *Content-based recommendation*
*Content-based recommendation* merupakan sebuah sistem rekomendasi yang menggunakan kesamaan fitur sebagai dasar rekomendasi. Sistem rekomendasi ini merekomendasikan item yang mirip dengan yang direferensikan pengguna di masa lalu. 

    Kategori item yang mirip dalam kasus proyek ini dapat direferensikan dari *genre* film. Selain *genre* film, fitur lain seperti *movieId*, *title*, dan *year* dibutuhkan untuk membantu identifikasi film. Agar kondisi diatas terpenuhi, maka dataset yang dibutuhkan pada sistem rekomendasi ini adalah dataset yang berisi informasi mengenai fitur *genre*, *movieId*, *title*, dan *year* dari dataframe *df*. Informasi-informasi tersebut akan dikumpulkan menjadi satu dataset baru bernama *title_genres*.
    
    Setelah membuat dataset *title_genres*, fitur *genres* juga diperlukan untuk pembuatan vektor yang akan dijadikan preferensi. Sehingga dengan perulangan, kumpulan fitur *genres* akan dimasukkan satu per satu ke dalam variabel *genres*. Saat dimasukkan ke dalam variabel *genres*, nilai dari fitur *genres* yang bertanda '|' akan digantikan dengan *whitespace*. Hasil variabel *genres* dapat dilihat pada tabel 16.

    Tabel 16. Variabel *genres* yang akan dijadikan vektor preferensi.
    | # | genres                                      |
    |:-:|---------------------------------------------|
    | 0 | Adventure Animation Children Comedy Fantasy |
    | 1 | Comedy Romance                              |
    | 2 | Action Crime Thriller                       |
    | 3 | Mystery Thriller                            |
    | 4 | Crime Mystery Thriller                      |
    
    Berdasarkan tabel 16, maka dapat disimpulkan bahwa dataset untuk *content-based recommendation* telah siap. Langkah selanjutnya adalah *preprocessing* untuk *collaborative recommendation*.

3. *Collaborative recommendation*
*Collaborative recommendation* merupakan sistem rekomendasi yang menggunakan pendapat komunitas atau pengguna terhadap suatu produk sebagai dasar rekomendasi. Sistem rekomendasi ini akan merekomendasikan produk yang kira-kira disukai pengguna lain berdasarkan pendapat pengguna lain dengan kesukaan yang sama. 

    Dataset yang dibutuhkan dalam sistem rekomendasi ini adalah dataset yang memuat informasi mengenai *rating* film serta beberapa fitur seperti *userId* dan *movieId*. Tahap pertama yang dilakukan untuk memenuhi kondisi tersebut adalah membuat dataset baru yang bernama *collaborative* yang berisi id user, id movie, dan rating.
    
    Setelah membuat dataset *collaborative*, masing-masing fitur *userId* dan *movieId* dilakukan pengkodean. Pengkodean ini dilakukan dengan cara mengambil nilai unik kedua fitur yang telah disebutkan kemudian dibuat sebuah *dictionary* untuk menampung nilai hasil pengkodean. Pengkodean ini dilakukan dengan tujuan membuat label untuk fitur *userId* dan *movieId*.
    
    Setelah pengkodean berhasil dilakukan, hasil pengkodean akan dimasukkan ke dalam dataset *collaborative* dengan menggunakan fungsi *map*. Untuk masing-masing hasil pengkodean *userId* dan *movieId* dimasukkan ke dalam kolom baru, yaitu *user* dan *movie*. Setelah melakukan hal tersebut, maka dilakukan pengecekan terhadap ukuran dari hasil pengkodean *userId* dan *movieId* serta dilakukan pengecekan terhadap minimum dan maksimum *rating* yang ada. Hasil pengecekan tersebut secara terurut adalah 610, 2269, 0.5, dan 5.
    
    Langkah selanjutnya adalah melakukan pengacakan sampel dataset *collaborative*. Hal ini dilakukan demi menghindari kluster sampel yang dominan. Pengacakan sampel dataset *collaborative* dilakukan dengan cara memanggil fungsi *sample*. Hasil pengacakan sampel dapat dilihat pada tabel 17.
    
     Tabel 17. Hasil pengacakan sampel dataset *collaborative*
    |   #   | userId | movieId | rating | user | movie |
    |:-----:|--------|---------|--------|------|-------|
    | 52039 | 599    | 1374    | 3.0    | 204  | 926   |
    | 16228 | 367    | 3809    | 5.0    | 131  | 209   |
    | 71258 | 45     | 2311    | 4.0    | 15   | 1529  |
    | 65637 | 522    | 4734    | 3.5    | 178  | 1338  |
    | 67623 | 282    | 63082   | 4.0    | 104  | 1403  |
    | ...   | ...    | ...     | ...    | ...  | ...   |
    | 6303  | 391    | 1089    | 5.0    | 140  | 57    |
    | 56237 | 484    | 74789   | 4.0    | 169  | 1033  |
    | 82651 | 285    | 3262    | 3.0    | 504  | 1963  |
    | 860   | 17     | 110     | 4.5    | 4    | 7     |
    | 15913 | 290    | 3671    | 4.0    | 107  | 203   |
    
    Berdasarkan tabel 17, dapat terlihat bahwa dataset *collaborative* sekarang telah memiliki sampel acak. Hal tersebut dapat terlihat melalui kolom *#* atau kolom *index*.
    
    Langkah selanjutnya adalah melakukan pembagian data. Data sampel dalam pembagian data ini adalah kolom *user* dan *movie*. Data sampel tersebut kemudian akan ditampung dalam sebuah variabel bernama *x*. Data target dalam pembagian data ini adalah kolom *rating* yang telah dinormalisasi menggunakan teknik *MinMax Normalization*. Data target tersebut kemudian akan ditampung dalam sebuah variabel bernama *y*. Hasil pembuatan variabel *x* dan *y* dapat dilihat pada gambar 5.
    
    ![collaborative_data.jpg](https://drive.google.com/uc?export=view&id=1Veg-OrxkZ9yf6OQnaoEHg8GE6OAj-xXU "collaborative_data")
    Gambar 5. Hasil pembuatan variabel *x* dan *y*
    
    Berdasarkan gambar 5, maka dapat terlihat bahwa susunan *list* pertama adalah variabel *x* dan susunan *list* kedua adalah variabel *y*.
    
    Setelah membuat data sampel dan data target, maka langkah selanjutnya adalah membagi data dengan rasio 90:10. Pembagian data tersebut dilakukan dengan cara memasukkan 90% data sampel dan data target ke variabel *x_train* dan *y_train* dan memasukkan 10% data sampel dan data target ke variabel *x_val* dan *y_val*.
    
    Akhirnya, tahap *data preparation* telah selesai, tahap selanjutnya adalah *modelling*.
    
## Modelling
Pada tahap ini akan dibuat fungsi dan *class* yang berperan sebagai model untuk sistem rekomendasi yang akan diterapkan dalam proyek ini. Masing-masing fungsi dan *class* tersebut akan menggunakan dataset khusus yang telah dibuat dan disiapkan pada tahap *data preparation*. Langkah pertama adalah membuat fungsi yang berperan sebagai model untuk *non-personalized recommendation*.

### *Non-personalized recommendation*
Pada *non-personalized recommendation*, akan dibuat sebuah fungsi bernama *popular_movies* yang berperan sebagai model untuk sistem rekomendasi ini. Fungsi tersebut memiliki dua parameter, yaitu data input dan jumlah sampel yang ingin dikembalikan. Secara garis besar, fungsi ini akan mengembalikan sebuah data yang memiliki kepopularitasan tertinggi. Untuk menghitung nilai popularitas, dibuat sebuah fungsi bernama *weighted_rate*. Fungsi ini dibuat berdasarkan referensi dari cara IMDB menentukan kepopularitasan [4]. Rumus dari fungsi tersebut adalah sebagai berikut.

> popularity = $$((vxR)+(mxC)) \over (v+m)$$

Berikut penjelasan dari fungsi tersebut.
- R = rata-rata *rating* per film
- v = jumlah frekuensi *rating* film
- m = batas minimum *v*
- C = rata-rata keseluruhan *R*

Secara detail, cara fungsi *popular_movies* bekerja adalah sebagai berikut.
- Pertama, fungsi ini akan membuat dua variabel baru yang berisi mengenai jumlah frekuensi *rating* film dan rata-rata *rating* per film.
- Kemudian dua variabel tersebut digabung menggunakan fungsi *merge* menjadi satu dataset yang bernama *popularMovies*
- Setelah itu fungsi *weighted_rate* dengan rumus di atas dibuat
- Kemudian, variabel-variabel yang dibutuhkan untuk fungsi *weighted_rate* dibuat. Variabel *R* dibuat dengan cara mengambil data pada kolom *averageRatings* pada dataset *popularMovies*. Variabel *v* dibuat dengan cara mengambil data pada kolom *numberOfVotes* pada dataset *popularMovies*. Variabel *m* dibuat dengan cara mengambil data di atas persentil 90 pada kolom *numberOfVotes*. Terakhir variabel *C* dibuat dengan cara menghitung rata-rata keseluruhan data pada kolom *averageRatings*.
- Setelah seluruh variabel yang dibutuhkan oleh fungsi *weighted_rate* dibuat, maka langkah selanjutnya adalah menghitung nilai kepopularitasan yang hasilnya akan dimasukkan ke kolom baru bernama *popularity* pada dataset *popularMovies*.
- Langkah selanjutnya adalah menyortir dataset *popularMovies* berdasarkan kolom *popularity*.
- Langkah terakhir adalah mengembalikan nilai dataset *popularMovies* dengan jumlah data sesuai dengan argumen yang dikirim sewaktu pemanggilan fungsi.

Hasil daftar rekomendasi dari fungsi *popular_movies* dapat dilihat pada tabel 18.

Tabel 18. Daftar rekomendasi *non-personalized recommendation*
| # | title                                             | numberOfVotes | averageRatings | popularity |
|:-:|---------------------------------------------------|---------------|----------------|------------|
| 0 | Elite Squad                                       | 4             | 4.25           | 3.489816   |
| 1 | His Girl Friday                                   | 4             | 4.25           | 3.489816   |
| 2 | Living in Oblivion                                | 4             | 4.25           | 3.489816   |
| 3 | Fog of War: Eleven Lessons from the Life of Ro... | 4             | 4.25           | 3.489816   |
| 4 | Creature Comforts                                 | 4             | 4.25           | 3.489816   |
| 5 | Persepolis                                        | 4             | 4.25           | 3.489816   |
| 6 | Seven Pounds                                      | 4             | 4.25           | 3.489816   |
| 7 | Paths of Glory                                    | 4             | 4.25           | 3.489816   |
| 8 | Rope                                              | 4             | 4.25           | 3.489816   |
| 9 | All Quiet on the Western Front                    | 4             | 4.25           | 3.489816   |

Kelebihan fungsi tersebut adalah kesederhanaan dalam menerapkan algoritma fungsi. Hal tersebut dikarenakan fungsi tersebut tidak memerlukan preferensi khusus pengguna. Kekurangan dari algortima fungsi ini adalah akan menghasilkan daftar rekomendasi yang sama bagi seluruh pengguna. Hal tersebut wajar karena tujuan dari diciptakannya fungsi ini adalah sebagai model untuk *non-personalized recommendation*.

Tahap *modelling* pada *non-personalized recommendation* telah selesai dilakukan. Tahap selanjutnya adalah *modelling* pada *content-based recommendation*.

## *Content-based recommendation*
Sistem rekomendasi ini dibangun dengan menggunakan teknik *count vectorizer*. Teknik tersebut digunakan untuk menemukan representasi fitur penting berdasarkan frekuensi kemunculan nilai terbanyak. Setelah representasi fitur dan korelasi antar fitur sudah diketahui, akan dicari derajat kesamaan antar fitur menggunakan teknik *cosine similarity*. Hal tersebut diwujudkan dengan menciptakan sebuah fungsi bernama *content_based_genre_by_user* yang akan berperan sebagai model bagi *content-based recommendation*.

Sebelum membuat fungsi tersebut, terdapat satu hal yang perlu dipersiapkan, yaitu membuat vektor fitur *genres*. Pembuatan vektor ini menggunakan *library CountVectorizer* yang berasal dari *sklearn* API. Hal pertama yang dilakukan untuk membuat vektor dengan *library* tersebut adalah membuat *instance* dari *CountVectorizer*. Kemudian dilakukan proses pembuatan vektor dengan memanggil fungsi *fit*. Setelah proses pembuatan vektor berhasil dilakukan, langkah terakhir adalah menyimpan hasil vektor ke dalam dataset baru bernama *count_table*. Dataset *count_table* dapat dilihat pada tabel 19.

Tabel 19. Dataset *count_table* yang berisi vektor fitur *genres*
|          title         | action | adventure | animation | ... | thriller | war | western |
|:----------------------:|--------|-----------|-----------|-----|----------|-----|---------|
| Toy Story              | 0      | 1         | 1         | ... | 0        | 0   | 0       |
| Grumpier Old Men       | 0      | 0         | 0         | ... | 0        | 0   | 0       |
| Heat                   | 1      | 0         | 0         | ... | 1        | 0   | 0       |
| Seven                  | 0      | 0         | 0         | ... | 1        | 0   | 0       |
| Usual Suspects, The    | 0      | 0         | 0         | ... | 1        | 0   | 0       |
| ...                    | ...    | ...       | ...       | ... | ...      |     |         |
| John Carter            | 1      | 1         | 0         | ... | 0        | 0   | 0       |
| Amityville Horror, The | 0      | 0         | 0         | ... | 1        | 0   | 0       |
| Christine              | 0      | 0         | 0         | ... | 0        | 0   | 0       |
| Eraserhead             | 0      | 0         | 0         | ... | 0        | 0   | 0       |
| Queen of the Damned    | 0      | 0         | 0         | ... | 0        | 0   | 0       |

Berdasarkan tabel 19, dapat terlihat bahwa seluruh fitur *genres* telah terpetakan menjadi vektor angka antara 1 dan 0. Angka 1 menunjukkan bahwa film tersebut memiliki fitur *genres* yang bersangkutan dan angka 0 menunjukkan bahwa film tersebut tidak memiliki fitur *genres* yang bersangkutan.

Setelah membuat vektor fitur *genres*, langkah selanjutnya adalah membuat fungsi bernama *content_based_genre_by_user* yang akan berperan sebagai model untuk *content-based recommendation*. Sesuai dengan nama dari fungsi tersebut, algoritma fungsi akan menentukan daftar rekomendasi berdasarkan fitur *genres*. Secara garis besar, algoritma fungsi ini bekerja dengan menggunakan teknik *cosine similarity*. Menurut Rahutomo *et al* [5], teknik *cosine similarity* merupakan teknik yang sering digunakan pada bidang *information retrieval* dengan mengukur derajat kesamaan antar dokumen. Penggunaan teknik ini juga sering digunakan pada bidang sistem rekomendasi, terutama untuk sistem rekomendasi yang menggunakan teknik *content-based filtering*. Hal tersebut terjadi karena teknik tersebut diharuskan menemukan kesamaan preferensi pengguna di masa lalu untuk menghasilkan daftar rekomendasi.

Secara detail, cara fungsi *content_based_genre_by_user* bekerja adalah sebagai berikut.
- Pertama, fungsi akan membuat sebuah variabel bernama *by_time* yang berisi informasi mengenai tiga film teratas yang diberi *rating* oleh pengguna. Nilai pengguna berasal dari argumen *userId* yang dikirim sewaktu fungsi ini terpanggil.
- Kemudian, fungsi akan membuat sebuah variabel lain bernama *user_pref* yang merupakan penggabungan antara dataset *df* dan *by_time*. Hasil akhir dari penggabungan ini adalah dataset yang berisi informasi mengenai seluruh fitur dataset *df* serta vektor *genres*.
- Setelah itu, fungsi akan membuat variabel lain bernama *total_user_pref* yang berisi mengenai jumlah preferensi *genres* yang disukai oleh pengguna. Pembuatan variabel ini dilakukan melakukan *grouping* berdasarkan argumen *userId*. Setelah dilakukan *grouping*, seluruh fitur *genres* dijumlahkan dengan menggunakan fungsi *sum*.
- Langkah selanjutnya, dengan menggunakan perulangan, dicari derajat kesamaan antara *total_user_pref* dan *count_table* dengan menggunakan *library cosine_similarity*. Hasil perhitungan derajat kesamaan kemudian dimasukkan ke dalam *dictionary* bernama *similarity*.
- Langkah terakhir adalah mengembalikan 10 data dari dataset *title_genres* yang berada dalam *dictionary similarity*. Data diambil dari dataset *title_genres* karena dataset yang digunakan untuk membuat vektor *genres* adalah dataset *title_genres*.

Hasil daftar rekomendasi dari fungsi *content_based_genre_by_user* dapat dilihat pada tabel 20.

Tabel 20. Daftar rekomendasi *content-based recommendation* dengan id pengguna 125.    
|   #  | movieId | title                              | genres                                     | year |
|:----:|---------|------------------------------------|--------------------------------------------|------|
| 814  | 48774   | Children of Men                    | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2006 |
| 831  | 91500   | The Hunger Games                   | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2012 |
| 956  | 5944    | Star Trek: Nemesis                 | Action\|Drama\|Sci-Fi\|Thriller            | 2002 |
| 972  | 8361    | Day After Tomorrow, The            | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2004 |
| 1045 | 87232   | X-Men: First Class                 | Action\|Adventure\|Sci-Fi\|Thriller\|War   | 2011 |
| 1047 | 88140   | Captain America: The First Avenger | Action\|Adventure\|Sci-Fi\|Thriller\|War   | 2011 |
| 1057 | 103772  | Wolverine, The                     | Action\|Adventure\|Fantasy\|Sci-Fi         | 2013 |
| 1391 | 58025   | Jumper                             | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2008 |
| 1436 | 117529  | Jurassic World                     | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2015 |
| 1698 | 136864  | Batman v Superman: Dawn of Justice | Action\|Adventure\|Fantasy\|Sci-Fi         | 2016 |

Kelebihan fungsi *content_based_genre_by_user* adalah menghasilkan daftar rekomendasi yang spesifik dan dinamis tergantung pengguna mana yang akan diberikan daftar rekomendasi. Hal tersebut terjadi karena algoritma fungsi ini dibangun berdasarkan derajat kesamaan preferensi pengguna di masa lalu. Kekurangan fungsi *content_based_genre_by_user* adalah pembangunan fungsi yang kompleks, karena membutuhkan data preferensi pengguna di masa lalu.

Tahap *modelling* pada *content-based recommendation* telah selesai dilakukan. Tahap selanjutnya adalah *modelling* pada *collaborative recommendation*.

## *Collaborative recommendation*
Sistem rekomendasi ini dibangun dengan membuat sebuah *class* bernama *CollRecNet* yang berperan sebagai model untuk *collaborative recommendation*. Model ini dibangun dengan mewarisi *library tf.keras.model*. Tujuan dari diciptakannya model ini adalah untuk menghitung skor kecocokan pengguna dan film menggunakan teknik *embedding*. Secara garis besar, model ini bekerja dengan cara melakukan proses *embedding* pada fitur *user* dan *movies*. Kemudian, dilakukan proses perkalian *dot product* antara *embedding user* dan *movies*.

Model *CollRecNet* mengimplementasikan dua metode yang diwarisi dari *tf.keras.model*. Metode pertama adalah adalah metode *\_init\_*. Metode tersebut bertujuan untuk melakukan inisialisasi properti pada model. Properti-properti yang dinisialisasi adalah *num_user*, *num_movies*, *embedding_size*, *users_embedding*, *user_bias*, *movies_embedding*, dan *movies_bias*. Properti *num_users* menunjukkan jumlah pengguna yang akan melalui proses *embedding*. Properti *num_movies* menunjukkan jumlah film yang akan melalui proses *embedding*. Properti *embedding_size* menunjukkan ukuran proses *embedding*. Properti *users_embedding* menunjukkan proses *embedding* untuk pengguna. Properti *user_bias* menunjukkan bias dari proses *embedding* untuk pengguna. Properti *movies_embedding* menunjukkan proses *embedding* untuk film. Terakhir properti *movies_bias* menunjukkan bias dari proses *embedding* untuk film.

Metode kedua adalah metode *call*. Metode ini merupakan metode inti model *CollRecNet*. Proses pelatihan model dilakukan dengan memanggil metode ini. Cara metode ini adalah sebagai berikut.
- Melakukan proses *embedding* pada *user* dan *movie* yang nilai vektornya disimpan ke dalam variabel *user_vector* dan *movies_vector*
- Bias dari proses *embedding* masing-masing akan disimpan dalam variabel *user_bias* dan *movies_bias*
- Menghitung *dot product* dari *user_vector* dan *movies_vector*. Hasil *dot product* kemudian disimpan ke dalam variabel *dot_user_movies*
- Menghitung nilai *weight* dengan menjumlahkan variabel *dot_user_movies*, *user_bias*, dan *movies_bias*. Nilai *weight* kemudian disimpan ke dalam variabel *x*
- Mengembalikan nilai *weight* yang telah diaktivasi menggunakan fungsi aktivasi *sigmoid*.

Setelah selesai pembuatan model *CollRecNet*, model kemudian di-*compile*. Parameter yang digunakan pada proses kompilasi model ada tiga. Pertama adalah *loss*. Fungsi *loss* yang digunakan pada model ini adalah *BinaryCrossEntropy*. Kedua adalah *optimizer*. *Optimizer* yang digunakan adalah *Adam* dengan *learning rate* sebesar 0,001. Terakhir adalah *metrics*. Matriks performa yang digunakan pada model ini adalah *RootSquareMeanError*.

Seusai model selesai melakukan proses kompilasi, model kemudian melalui proses *fitting* atau proses pelatihan. Proses pelatihan ini dilakukan dengan *epochs* sebanyak 20 dan *batch_size* sebanyak 8. Hasil daftar rekomendasi dari model *CollRecNet* dapat dilihat pada tabel 21.

Tabel 21. Daftar rekomendasi *collaborative recommendation* dengan id pengguna 474.
| # | title                                  | genres                                        | year |
|:-:|----------------------------------------|-----------------------------------------------|------|
| 0 | Bound                                  | Crime\|Drama\|Romance\|Thriller               | 2006 |
| 1 | High Plains Drifter                    | Western                                       | 1973 |
| 2 | Drunken Master (Jui kuen)              | Action\|Comedy                                | 1978 |
| 3 | Hedwig and the Angry Inch              | Comedy\|Drama\|Musical                        | 2000 |
| 4 | Fantastic Mr. Fox                      | Adventure\|Animation\|Children\|Comedy\|Crime | 2009 |
| 5 | Intouchables                           | Comedy\|Drama                                 | 2011 |
| 6 | Louis C.K.: Live at the Beacon Theater | Comedy                                        | 2011 |
| 7 | Hunt, The (Jagten)                     | Drama                                         | 2012 |
| 8 | Dallas Buyers Club                     | Drama                                         | 2013 |
| 9 | Nightcrawler                           | Crime\|Drama\|Thriller                        | 2014 |

Akhirnya, tahap *modelling* pada *ollaborative recommendation* telah selesai dilakukan. Tahap selanjutnya adalah evaluasi ketiga sistem rekomendasi yang dibangun pada proyek ini.

## Evaluation
Pada tahap ini, akan dilakukan evaluasi terhadap ketiga sistem rekomendasi yang telah dibuat pada proyek ini. Evaluasi pertama yang dilakukan adalah evaluasi pada *non-personalized recommendation*.

### *Non-personalized recommendation*
Evaluasi pada sistem rekomendasi ini hanya melakukan analisis terhadap daftar rekomendasi yang dihasilkan melalui fungsi *populer_movies*. Pada sistem rekomendasi ini, tidak dilakukan evaluasi menggunakan matriks performansi dikarenakan daftar rekomendasi yang dikeluarkan akan tetap sama siapapun penggunanya. Hasil daftar rekomendasi yang dikeluarkan oleh fungsi *popular_movies* dengan *userId* 125 dapat dilihat pada tabel 22.

Tabel 22. Daftar rekomendasi *non-personalized recommendation*
| # | title                                             | numberOfVotes | averageRatings | popularity |
|:-:|---------------------------------------------------|---------------|----------------|------------|
| 0 | Elite Squad                                       | 4             | 4.25           | 3.489816   |
| 1 | His Girl Friday                                   | 4             | 4.25           | 3.489816   |
| 2 | Living in Oblivion                                | 4             | 4.25           | 3.489816   |
| 3 | Fog of War: Eleven Lessons from the Life of Ro... | 4             | 4.25           | 3.489816   |
| 4 | Creature Comforts                                 | 4             | 4.25           | 3.489816   |
| 5 | Persepolis                                        | 4             | 4.25           | 3.489816   |
| 6 | Seven Pounds                                      | 4             | 4.25           | 3.489816   |
| 7 | Paths of Glory                                    | 4             | 4.25           | 3.489816   |
| 8 | Rope                                              | 4             | 4.25           | 3.489816   |
| 9 | All Quiet on the Western Front                    | 4             | 4.25           | 3.489816   |

Berdasarkan tabel 22, dapat terlihat bahwa terdapat *list* film yang terurut berdasarkan kolom *popularity*. Dapat dilihat juga rata-rata *rating* terhadap film-film tersebut adalah 4.25, rata-rata *rating* yang sangat tinggi.

Tahap evaluasi pada *non-personalized recommendation* telah selesai dilakukan. Tahap evaluasi selanjutnya adalah evaluasi pada *content-based recommendation*.

### *Content-based recommendation*
Evaluasi pada sistem rekomendasi ini terdiri dari dua tahap, yaitu analisis daftar rekomendasi yang dikeluarkan oleh fungsi *content_based_genre_by_user* dan evaluasi matriks performansi. Matriks performansi yang digunakan untuk mengevaluasi performa sistem rekomendasi ini adalah *Normalize Discount Cumulative Gain*.

1. **Daftar rekomendasi**
Daftar rekomendasi pada *content-based recommendation* dihasilkan dari fungsi bernama *content_based_genre_by_user*. Fungsi tersebut menghasilkan sebuah daftar rekomendasi dengan cara mencari derajat kesamaan preferensi pengguna di masa lalu. Preferensi fitur yang digunakan pada fungsi tersebut adalah fitur *genres*. Daftar rekomendasi yang dihasilkan oleh fungsi *content_based_genre_by_user* dapat dilihat pada tabel 23.

    Tabel 23. Daftar rekomendasi *content-based recommendation* dengan id pengguna adalah 125.
    |   #  | movieId | title                              | genres                                     | year |
    |:----:|---------|------------------------------------|--------------------------------------------|------|
    | 814  | 48774   | Children of Men                    | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2006 |
    | 831  | 91500   | The Hunger Games                   | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2012 |
    | 956  | 5944    | Star Trek: Nemesis                 | Action\|Drama\|Sci-Fi\|Thriller            | 2002 |
    | 972  | 8361    | Day After Tomorrow, The            | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2004 |
    | 1045 | 87232   | X-Men: First Class                 | Action\|Adventure\|Sci-Fi\|Thriller\|War   | 2011 |
    | 1047 | 88140   | Captain America: The First Avenger | Action\|Adventure\|Sci-Fi\|Thriller\|War   | 2011 |
    | 1057 | 103772  | Wolverine, The                     | Action\|Adventure\|Fantasy\|Sci-Fi         | 2013 |
    | 1391 | 58025   | Jumper                             | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2008 |
    | 1436 | 117529  | Jurassic World                     | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2015 |
    | 1698 | 136864  | Batman v Superman: Dawn of Justice | Action\|Adventure\|Fantasy\|Sci-Fi         | 2016 |
    
    Berdasarkan tabel 23, maka dapat terlihat bahwa film-film yang direkomendasikan di atas memiliki kemiripan *genre* yang identik. Sehingga berdasarkan tabel 23 dapat disimpulkan bahwa pengguna dengan id 125 cukup menyukai film yang memiliki *genre action*, *adventure*, dan *thriller*.
    
    Selanjutnya akan dilakukan evaluasi *content-based recommendation* menggunakan matriks evaluasi *Normalize Discount Cumulative Gain*.

2. **_Normalize Discount Cumulative Gain (NDCG)_**
*Normalize Discount Cumulative Gain* atau disingkat NDCG merupakan sebuah matriks evaluasi yang sering digunakan untuk melakukan evaluasi terhadap *top-n list* [6]. NDCG bekerja dengan cara membandingkan dua data berbentuk *top-n list* yang dihasilkan dari model yang sama dengan fitur tertentu. Hasil perbandingan kedua data tersebut akan menunjukkan seberapa bagus model dalam membuat *top-n list*. Semakin tinggi hasil perbandingan NDCG maka semakin bagus model yang dibuat dalam menyajikan *top-n list*. NDCG dapat didefinisikan melalui persamaan berikut.
    > NDCG = $$data1 \over data2$$, where data1 and data2 is $$\sum_{i=1}^{N}$$ $$data[i] \over log_2(i+1)$$
    
    Sebelum melakukan melakukan evaluasi menggunakan NDCG, akan dilakukan pembuatan fungsi lain yang mirip dengan cara kerja algoritma fungsi *content_based_genre_by_user*. Fungsi tersebut bernama *get_movie_preference_by_user*. Tujuan diciptakannya fungsi tersebut adalah untuk menciptakan daftar rekomendasi yang berbeda sebagai pembanding untuk daftar rekomendasi yang dikeluarkan oleh fungsi *content_based_genre_by_user*. 
    
    Cara kerja fungsi *get_movie_preference_by_user* sangat identik mulai dari pembuatan variabel *by_time* hingga hasil kembalian fungsi yang berupa *top-10 recommendation*. Namun, terdapat perbedaan diantara dua fungsi tersebut. Perbedaan tersebut adalah isi dari variabel *by_time*. Isi variabel *by_time* pada fungsi *content_based_genre_by_user* berisi tiga film teratas yang diberi *rating* oleh pengguna yang bersangkutan, sedangkan fungsi *get_movie_preference_by_user* berisi *ranking* keempat hingga keenam film teratas yang diberi *rating* oleh pengguna yang bersangkutan. Jika fungsi *get_movie_preference_by_user* dijalankan, maka akan menghasilkan daftar film yang memiliki genre mirip dengan daftar rekomendasi hasil *content_based_genre_by_user*. Daftar rekomendasi yang dihasilkan oleh fungsi *get_movie_preference_by_user* dapat dilihat pada tabel 24.
    
    Tabel 24. Daftar rekomendasi *content-based recommendation* dengan id pengguna adalah 125 menggunakan fungsi *get_movie_preference_by_user*.
    |   #  | movieId | title                                        | genres                          | year |
    |:----:|---------|----------------------------------------------|---------------------------------|------|
    | 593  | 5378    | Star Wars: Episode II - Attack of the Clones | Action\|Adventure\|Sci-Fi\|IMAX | 2002 |
    | 616  | 8636    | Spider-Man 2                                 | Action\|Adventure\|Sci-Fi\|IMAX | 2004 |
    | 719  | 72998   | Avatar                                       | Action\|Adventure\|Sci-Fi\|IMAX | 2009 |
    | 732  | 95510   | Amazing Spider-Man, The                      | Action\|Adventure\|Sci-Fi\|IMAX | 2012 |
    | 830  | 89745   | Avengers, The                                | Action\|Adventure\|Sci-Fi\|IMAX | 2012 |
    | 838  | 110102  | Captain America: The Winter Soldier          | Action\|Adventure\|Sci-Fi\|IMAX | 2014 |
    | 1054 | 102445  | Star Trek Into Darkness                      | Action\|Adventure\|Sci-Fi\|IMAX | 2013 |
    | 1887 | 82461   | Tron: Legacy                                 | Action\|Adventure\|Sci-Fi\|IMAX | 2010 |
    | 1896 | 106002  | Ender's Game                                 | Action\|Adventure\|Sci-Fi\|IMAX | 2013 |
    | 2236 | 93363   | John Carter                                  | Action\|Adventure\|Sci-Fi\|IMAX | 2012 |
    
    Berdasarkan tabel 24, maka dapat terlihat bahwa daftar rekomendasi film yang dihasilkan memiliki *genre* yang mirip dengan daftar rekomendasi tabel 23, namun memiliki judul-judul yang berbeda. Berdasarkan tabel 24 juga dapat disimpulkan bahwa evaluasi NDCG telah siap dilakukan.
    
    Setelah pembuatan fungsi *get_movie_preference_by_user* berhasil dilakukan, langka selanjutnya adalah adalah membuat fungsi *ndcg_result*. Algoritma fungsi tersebut sangatlah sederhana. Cara fungsi tersebut bekerja yaitu pertama, fungsi tersebut akan memanggil fungsi *content_based_genre_by_user* yang data hasil keluaran fungsi akan dimasukkan ke dalam variabel *rec_1*. Setelah itu, fungsi tersebut akan memanggil fungsi *get_movie_preference_by_user* yang data hasil keluaran fungsi akan dimasukkan ke dalam variabel *rec_2*. Langkah terakhir adalah melakukan perhitungan skor NDCG dengan menggunakan *library ndcg_score* yang berasal dari *sklearn* API.
    
    Terdapat tiga percobaan yang dilakukan untuk mengecek skor NDCG *content-based recommendation*. Tiga percobaan tersebut dilakukan pada pengguna dengan id 125, 45, dan 5. Skor NDCG dari ketiga percobaan secara terurut adalah 0.940545902862712, 0.8286799596508156, dan 0.9030449155550614. Dari ketiga percobaan tersebut, skor yang paling kecil adalah 0.8286799596508156 sehingga dapat disimpulkan *content-based recommendation* yang dibuat pada proyek ini cukup mengesankan.
    
### *Collaborative recommendation*
Evaluasi pada sistem rekomendasi ini terdiri dari dua tahap, yaitu visulasisai matriks performansi dan analisis daftar rekomendasi. Matriks performansi yang digunakan untuk mengevaluasi performa sistem rekomendasi ini adalah *Root Mean Square Error*.

1. **_Root Mean Square Error (RMSE)_**
*Root Mean Squared Error* atau disingkat RMSE merupakan salah satu algoritma yang sering digunakan pada permasalahan prediksi. RMSE juga sering digunakan sebagai matriks evaluasi sistem rekomendasi menggunakan teknik *collaborative filtering*. RMSE sering digunakan pada sistem rekomendasi tersebut karena teknik *collaborative filtering* pada dasarnya juga merupakan kasus prediksi yang memprediksi *unseen product* yang kemungkinan disukai oleh pengguna. RMSE bekerja dengan cara menghitung akar dari jumlah selisih kuadrat rata-rata antara nilai sebenarnya dengan nilai yang diprediksi oleh model. Semakin rendah nilai RMSE maka semakin bagus performa model. RMSE dapat didefinisikan melalui persamaan berikut.
    > RMSE = $$\sqrt (\sum_{i=1}^N (Predicted_i - Actual_i)^2) \over \sqrt N$$

    Visualisasi matriks RMSE pada model *CollRecNet* dilakukan dengan menggunakan variabel *history* yang berisi rekaman matriks RMSE pada saat pelatihan. *Library* visualisasi yang digunakan adalah *library mathplotlib.pyplot*. Hasil visualisasi performa menggunakan RMSE dapat dilihat pada gambar 6.
    
    ![collaborative_metrics.jpg](https://drive.google.com/uc?export=view&id=1DZyrQcolRuOeR_6wUrQVfVTWqZS33jgt "collaborative_metrics")
    Gambar 6. Performa model *CollRecNet*
    
    Berdasarkan gambar 6, maka dapat disimpulkan bahwa proses pelatihan model cukup lancar dan model konvergen pada epochs sekitar 10. Berdasarkan gambar 6 juga dapat disimpulkan bahwa model memperoleh nilai error akhir sebesar sekitar 0.194 pada data latih dan error pada data validasi sebesar 0.183. Angka yang cukup memuaskan untuk performa *collaborative recommendation* pada proyek ini.

2. **Hasil rekomendasi**
Untuk mendapatkan hasil rekomendasi film dari model *CollRecNet*, pertama-tama perlu diambil sampel film yang belum pernah ditonton oleh pengguna. Pengambilan sampel tersebut dilakukan dengan cara memilih pengguna secara acak. Kemudian diambil data film yang pernah diberi *rating* oleh pengguna tersebut. Dengan memanfaatkan data film yang pernah diberi *rating* oleh pengguna tersebut, maka dapat dibuat data film yang belum pernah diberi *rating* oleh pengguna dengan cara memanfaatkan operator '~' dan fungsi *isin*. Setelah daftar film yang belum pernah ditonton telah didapatkan, langkah selanjutnya adalah menyortir daftar film tersebut menjadi bernilai unik. Kemudian daftar film yang belum pernah diberi *rating* tersebut akan dikonversi menjadi label film sesuai dengan id film dengan memanfaatkan *movie_to_movie_encoded*. Selain sampel film, label pengguna juga diperlukan sebagai input prediksi model. Pengkonversian id pengguna dilakukan dengan memanfaatkan *user_to_user_encode*. Setelah label pengguna dan label film dibuat, langkah selanjutnya adalah membuat *list* yang berisi gabungan dari kedua label tersebut. *List* inilah yang akan menjadi input prediksi model.

    Hasil rekomendasi didapatkan dari memanggil fungsi *predict* dengan hasil *list* label pengguna dan label film yang belum pernah diberi *rating* sebagai input. Namun, hasil rekomendasi ini masih belum dapat digunakan karena masih berbentuk label. Oleh karena itu, konversi dari label menjadi judul film perlu dilakukan. Konversi ini akan menggunakan variabel *movie_encoded_to_movie* sehingga menghasilkan sebuah data yang memiliki daftar rekomendasi judul film. Langkah selanjutnya adalah menampilkan daftar rekomendasi film tersebut. Sebagai perbandingan akan ditampilkan juga daftar film yang pernah diberi rating oleh pengguna dengan memanfaatkan variabel *movie_watched_by_user*. Hasil rekomendasi model *CollRectNet* dapat dilihat pada gambar 7.
    
    ![collaborative_result.jpg](https://drive.google.com/uc?export=view&id=1Pu0rWqs_IPOez8eJ6dwkQ80DDjdva_GG "collaborative_result")
    Gambar 7. Perbandingan *collaborative recommendation* dengan film yang pernah diberi *rating* oleh pengguna dengan id 474.
    
    Berdasarkan gambar 7, maka dapat terlihat bahwa daftar rekomendasi model *CollRecNet* dengan daftar film yang pernah diberi *rating* oleh pengguna cukup mirip. Hal tersebut terlihat dari *genre* film di kedua daftar film yang dominan hampir sama, yaitu genre *musical*, *drama*, dan *romance*.

## Kesimpulan
Berdasarkan proyek yang telah dilakukan, maka dapat disimpulkan beberapa hal sebagai berikut.
- Berdasarkan EDA, fitur *rating* dan *genre* cukup berpengaruh untuk dijadikan sebagai fitur preferensi sistem rekomendasi yang akan dibangun.
- Berdasarkan hasil evaluasi *non-personalized recommendation*, daftar rekomendasi yang dihasilkan cukup memuaskan terlihat dari rata-rata *rating* film dalam daftar tersebut.
- Berdasarkan hasil NDCG, daftar rekomendasi yang dihasilkan pada *content-based recommendation* cukup memuaskan.
- Berdasarkan hasil RMSE, daftar rekomendasi yang dihasilkan pada *collaborative recommendation* cukup memuaskan.
- Kesimpulan akhir, ketiga sistem rekomendasi telah berhasil dibuat.

> Proyek ini dapat diakses melalui url: https://colab.research.google.com/drive/1UpW-594QTGqrw5_Gin0go57RLfU4T1zd?usp=share_link

**Daftar Referensi**
- [1] Pazzani, M. J., & Billsus, D. (2007). *Content-based recommendation systems*. In The adaptive web (pp. 325-341). Springer, Berlin, Heidelberg.
- [2] Sarwar, B. M., Karypis, G., Konstan, J., and Riedl, J. (2002). *Recommender systems for large-scale e-commerce: Scalable neighborhood formation using clustering*. in Proceedings of the international conference on computer and information technology.
- [3] Shinigami. (2020). *Movie Recommender System Dataset*. https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset, diakses tanggal 27 November 2022.
- [4] Laksito, A. (2021). *Simple rekomendasi dengan Formula IMDb Weighted Rating*. https://blog.ariflaksito.net/2021/03/simple-rekomendasi-dengan-formula-imdb.html, diakses tanggal 27 November 2022.
- [5] Rahutomo, F., Kitasuka, T., & Aritsugi, M. (2012, October). *Semantic cosine similarity*. In The 7th international student conference on advanced science and technology ICAST (Vol. 4, No. 1, p. 1).
- [6] Wang, Y., Wang, L., Li, Y., He, D., & Liu, T. Y. (2013, June). *A theoretical analysis of NDCG type ranking measures*. In Conference on learning theory (pp. 25-54). PMLR.

**---Ini adalah bagian akhir dari laporan---**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [kaggle]: <https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset>