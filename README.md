# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech - mdavap

## Business Understanding
Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah beroperasi sejak tahun 2000 dan telah membangun reputasi yang kuat dalam menghasilkan lulusan berkualitas. Namun, institusi ini menghadapi tantangan signifikan berupa tingginya angka mahasiswa yang tidak menyelesaikan pendidikan (dropout), yang dapat berdampak negatif pada reputasi dan pendapatan institusi. Untuk mengatasi hal ini, institusi tersebut berupaya mengembangkan sistem deteksi dini berbasis data untuk mengidentifikasi mahasiswa yang berisiko dropout, sehingga dapat memberikan bimbingan khusus sebelum mereka benar-benar mengambil keputusan untuk berhenti.

Dengan begitu, pada proyek ini akan membuat sebuah dashboard untuk membantu Jaya Jaya Institut serta model Machine Learning agar bisa mempredikisi siswa yang akan mungkin melakukan dropout.

### Permasalahan Bisnis
Permasalahan bisnis utama yang perlu diselesaikan adalah tingginya tingkat dropout mahasiswa di Jaya Jaya Institut yang dapat mempengaruhi reputasi dan pendapatan institusi. Untuk mengatasi hal ini, institusi membutuhkan sistem deteksi dini berbasis machine learning yang dapat memprediksi dan mengidentifikasi mahasiswa yang berisiko dropout berdasarkan analisis data performa siswa. Model machine learning dapat membantu menganalisis pola-pola dari data historis untuk menghasilkan prediksi yang akurat. Selain itu, institusi juga memerlukan dashboard yang dapat memudahkan pemantauan dan pemahaman terhadap data performa siswa, sehingga pihak manajemen dapat mengambil tindakan preventif yang tepat dengan memberikan bimbingan khusus kepada mahasiswa yang teridentifikasi berisiko dropout berdasarkan hasil prediksi model.

### Cakupan Proyek
Cakupan proyek yang akan dikerjakan disini adalah pengembangan suatu dashboard untuk serta pembuatan model machine learning untuk memprediksi siswa yang mungkin akan keluar atau dropout.

### Persiapan

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance

Setup environment:
```
conda create -n sbm_meta python=3.10
conda activate sbm_meta
pip install -r requirements.txt
```
Running notebook:
```
jupyter .
```

## Machine Learning

### Data Preparation
- Dataset yang digunakan memiliki 37 kolom dan total data 4424.
- Dataset ini tidak memiliki data kosong maupun data yang terduplikasi, dapat disimpulkan data ini bersih.
- Selanjutnya, melakukan encoding pada kolom `Status` menjadi `StatusEncoded` supaya machine learning dapat berkerja dengan baik, aturan encoding sebagai berikut:
    - `Dropout` akan diubah menjadi `0`
    - `Graduate` akan diubah menjadi `1`
    - `Enrolled` akan diubah menjadi `2`
- Selanjutnya, melakukan normalisasi menggunakan `MinMaxScaler` pada data selain `Status` dan `StatusEncoded`.
- Selanjutnya, memilih siswa yang hanya keluar dari sekolah atau `Dropout` sama dengan `0.
- Dan yang terakhir membagi dataset sebesar 20% untuk data uji dan 80% untuk data latih

### Modeling
- **Logistic Regression**: Model klasifikasi linear dengan maksimum 1000 iterasi dan random state 99. Model ini bekerja dengan mencari hubungan linear antara fitur input dan probabilitas kelas output, cocok untuk klasifikasi biner dan multiclass dengan komputasi yang relatif ringan.
- **K-Nearest Neighbors**: Model yang mengklasifikasikan data berdasarkan mayoritas kelas dari k tetangga terdekat. Menggunakan parameter default yang berarti k=5 dan metric jarak Euclidean, cocok untuk dataset dengan pola lokal yang kuat.
- **Support Vector Machine**: Model yang bekerja dengan mencari hyperplane optimal untuk memisahkan kelas-kelas data. Menggunakan random state 99 untuk konsistensi, cocok untuk data dengan dimensi tinggi dan dapat menangani klasifikasi non-linear melalui kernel trick.
- **Decision Tree**: Model berbasis pohon keputusan dengan random state 44. Bekerja dengan membagi data berdasarkan fitur-fitur yang paling informatif, menghasilkan model yang mudah diinterpretasi namun rentan terhadap overfitting.
- **Random Forest**: Ensemble model yang terdiri dari 100 pohon keputusan dengan random state 88. Menggabungkan prediksi dari banyak pohon keputusan untuk menghasilkan prediksi yang lebih stabil dan akurat.
- **Gradient Boosting**: Model ensemble yang membangun pohon keputusan secara sekuensial dengan random state 99. Setiap pohon baru berfokus pada memperbaiki kesalahan prediksi dari pohon-pohon sebelumnya.
- **Naive Bayes**: Implementasi Multinomial Naive Bayes yang cocok untuk klasifikasi teks. Menggunakan teorema Bayes dengan asumsi independensi antar fitur, efisien untuk dataset dengan dimensi tinggi.
- **Multi-Layer Perceptron**: Neural network sederhana dengan maksimum 1000 iterasi dan random state 99. Mampu mempelajari pola non-linear yang kompleks melalui multiple hidden layers.
- **AdaBoost**: Model ensemble yang meningkatkan performa dengan memberikan bobot lebih pada data yang salah klasifikasi. Menggunakan random state 99 untuk reproduktifitas hasil.
- **Linear Discriminant Analysis**: Teknik klasifikasi yang mengasumsikan distribusi normal dan varians yang sama antar kelas. Bekerja dengan mereduksi dimensi sambil memaksimalkan separabilitas antar kelas.
- **Quadratic Discriminant Analysis**: Variasi dari LDA yang mengasumsikan setiap kelas memiliki matriks kovarians yang berbeda. Lebih fleksibel dari LDA namun memerlukan lebih banyak data.
- **Extra Trees**: Model ensemble mirip Random Forest namun dengan pemilihan split yang lebih acak. Menggunakan random state 99, biasanya menghasilkan model yang lebih cepat training-nya dibanding Random Forest.
- **Stochastic Gradient Descent**: Implementasi berbagai loss functions dengan optimisasi gradient descent stokastik. Menggunakan random state 99, efisien untuk dataset besar dengan update parameter per sampel.
- **Bagging Classifier**: Model ensemble yang melatih beberapa base classifier pada subset data yang berbeda. Menggunakan random state 99, efektif untuk mengurangi overfitting dan variance.

### Evaluation
Evaluasi disini akan menggunakan metrik Accuracy dan Confusion Matrix.
Berikut ini grafik evaluasi untuk setiap model:
![](https://i.imgur.com/iXoyIs5.png)

Evaluasi setiap model sebagai berikut:
- **Logistic Regression** (Akurasi: 76,2%)
Kelebihan: Menunjukkan performa yang seimbang dengan True Positive (211) dan True Negative (414) yang kuat. Tingkat False Positive relatif rendah (47), menunjukkan presisi yang baik dalam prediksi positif.
Kekurangan: Meskipun tingkat False Negative rendah (8), masih ada ruang untuk peningkatan dalam mengidentifikasi semua kasus positif dengan benar.
- **K-Nearest Neighbors** (Akurasi: 66,8%)
Kelebihan: Mempertahankan tingkat True Positive (186) dan True Negative (375) yang cukup baik meskipun akurasi keseluruhan lebih rendah.
Kekurangan: Tingkat False Positive (74) dan False Negative (44) lebih tinggi dibandingkan model lain, menunjukkan kinerja yang kurang optimal untuk dataset ini.
- **Support Vector Machine** (Akurasi: 74,0%)
Kelebihan: Tingkat True Negative sangat baik (429) dan False Negative sangat rendah (4), menunjukkan kemampuan kuat dalam mengidentifikasi kasus negatif.
Kekurangan: Tingkat False Positive (58) masih bisa ditingkatkan untuk meningkatkan akurasi prediksi secara keseluruhan.
- **Decision Tree** (Akurasi: 67,9%)
Kelebihan: Keseimbangan yang baik antara True Positive (197) dan False Positive yang relatif rendah (38).
Kekurangan: Tingkat False Negative lebih tinggi (33) dan True Negative lebih rendah (346) dibandingkan model lain, menunjukkan kemungkinan underfitting.
- **Random Forest** (Akurasi: 77,6%)
Kelebihan: Akurasi terbaik secara keseluruhan dengan tingkat True Positive (212) dan True Negative (420) yang kuat, menunjukkan keseimbangan yang sangat baik.
Kekurangan: Masih memiliki beberapa False Positive (46) dan False Negative (11), meskipun relatif rendah dibandingkan model lain.
- **Gradient Boosting** (Akurasi: 76,9%)
Kelebihan: Memiliki tingkat True Positive (210) dan True Negative (410) yang kuat dengan False Positive yang rendah (40).
Kekurangan: Tingkat False Negative (9) relatif rendah namun masih bisa ditingkatkan untuk performa yang lebih baik.
- **Naive Bayes** (Akurasi: 65,8%)
Kelebihan: Tingkat True Negative cukup baik (401) meskipun akurasi keseluruhan lebih rendah.
Kekurangan: Tingkat False Positive tertinggi (106) dan False Negative yang signifikan (42), menunjukkan kinerja klasifikasi yang kurang baik secara keseluruhan.
- **Multi-Layer Perceptron** (Akurasi: 72,8%)
Kelebihan: Keseimbangan yang baik antara True Positive (201) dan tingkat False Positive yang rendah (37).
Kekurangan: Tingkat True Negative lebih rendah (381) dibandingkan model dengan performa terbaik.
- **AdaBoost** (Akurasi: 75,5%)
Kelebihan: Tingkat True Positive (220) dan True Negative (410) yang kuat, menunjukkan keseimbangan yang baik secara keseluruhan.
Kekurangan: Tingkat False Positive (47) dan False Negative (14) masih bisa dioptimalkan untuk performa yang lebih baik.
- **Linear Discriminant Analysis** (Akurasi: 74,0%)
Kelebihan: Tingkat False Negative sangat rendah (0) dan True Negative yang kuat (409), menunjukkan identifikasi kasus negatif yang sangat baik.
Kekurangan: Tingkat False Positive (49) masih mempengaruhi kinerja keseluruhan.
- **Quadratic Discriminant Analysis** (Akurasi: 68,9%)
Kelebihan: Tingkat True Positive yang baik (187) dan True Negative yang cukup (380).
Kekurangan: Tingkat False Negative lebih tinggi (33) dan False Positive (57) yang masih perlu diperbaiki.
- **Extra Trees** (Akurasi: 77,4%)
Kelebihan: Kinerja keseluruhan yang kuat dengan tingkat True Positive (211) dan True Negative (422) yang tinggi.
Kekurangan: Masih menunjukkan beberapa False Positive (47) dan False Negative (9), meskipun relatif rendah.
- **Stochastic Gradient Descent** (Akurasi: 74,7%)
Kelebihan: Tingkat True Positive sangat baik (223) dan True Negative yang kuat (435).
Kekurangan: Tingkat False Positive yang masih cukup tinggi (62) dan False Negative (11) mempengaruhi presisi keseluruhan.
- **Bagging Classifier** (Akurasi: 74,5%)
Kelebihan: Tingkat True Positive yang baik (213) dan False Positive yang relatif rendah (41).
Kekurangan: Tingkat False Negative (20) dan True Negative (392) masih bisa ditingkatkan untuk performa yang lebih optimal.

Kesimpulan: Secara keseluruhan, Random Forest menunjukkan performa terbaik dengan akurasi 77,6% dan keseimbangan yang baik antara semua metrik dan oleh sebab itu Random Forest dipilih.


## Business Dashboard
Dashboard terdiri dari 3 bagian yaitu:
1. Top Section - Key Perfomance Indicators
    - Total Students (Number), Jumlah siswa.
    - Total Enrolled Students (Number), Jumlah siswa yang masuk.
    - Total Graduated Students (Number), Jumlah siswa yang telah lulus.
    - Total Dropout Students (Number), Jumlah siswa yang keluar.
    - Dropout Rate (Number), Keseluruhan presentase dari siswa yang keluar.
    - Avarage Grade at First Semester (Number), Rata-rata nilai dari siswa disemester satu.
    - Avarage Grade at Second Semester (Number), Rata-rata nilai dari siswa disemester dua.
2. Middle Section - Student Characteristics and Student Perfomances
    - Student Characteristics
        - Student Nationality (World Map), Peta dunia yang menunjukan dari mana saja siswa berasal.
        - Student Courses (Pie Chart), Grafik yang menunjukan mata pelajaran yang siswa ambil.
        - Previous Student Qualification (Pie Chart), Grafik yang menunjukan kualifikasi apa saja sebelum siswa masuk dalam institut ini.
        - Student Martial Status (Pie Chart), Grafik yang menunjukan status pernikahan dari siswa-siswa.
        - Student Parent Education (Pie Chart), Grafik yang menunjukan pendidikan terakhir dari siswa-siswa.
        - Student Attendance Time (Pie Chart), Grafik yang menunjukan pada waktu apa siswa menghadiri pembelajaran.
    - Student Perfomances
        - Avarage Student Perfomance on First Semester (Row Chart), Grafik yang menunjukan rata-rata performa atau nilai pada siswa-siswa disemester satu.
        - Avarage Student Perfomance on Second Semester (Row Chart), Grafik yang menunjukan rata-rata performa atau nilai pada siswa-siswa disemester dua.
3. Bottom Section - Student Dropout Characteristics
    - Top 10 Courses with Highest Dropout count (Row Chart), Grafik yang menunjukan 10 mata pembelajaran teratas dengan jumlah putus sekolah.
    - Total Dropout by Student Gender (Bar Chart), Grafik yang menunjukan berapa banyak siswa keluar dengan perbandingan jenis kelamin mereka.
    - Total Dropout by Student Debtor (Pie Chart), Grafik yang menunjukan berapa banyak siswa keluar dengan perbandingan kepemilikan hutang mereka.
    - Total Dropout by Student Scholarship Holder (Pie Chart), Grafik yang menunjukan berapa banyak siswa keluar dengan perbandingan kepemilikan beasiswa.
    - Total Dropout by Student has Tuition Fees Up To Date (Pie Chart), Grafik yang menunjukan berapa banyak siswa keluar dengan perbandingan berapa dari mereka yang harus melunasi pembayaran.

[Link Dashboard](https://ds-meta.getani.me/public/dashboard/0dde6513-bb4c-4485-ac66-18a09e9d2dd3)
dan Berikut username dan password untuk metabase
```
root@mail.com
root123
```

## Menjalankan Sistem Machine Learning
Cara untuk menjalankan sistem machine learning sebagai berikut:
```
streamlit run app.py
```
[Klik disini untuk membuka versi cloud yang sudah siap](https://ds-jaya-jaya-v2-fwsgwjnwyqzfd7r427mppu.streamlit.app/)

## Conclusion
- Kesimpulan:
    - Solusi mengatasi siswa dropout ada dua pendekatan yaitu:
        1. Pembuatan dashboard yang berguna untuk memonitor yang mungkin menjadi faktor siswa drop out.
        2. Menggunakan model machine learning yaitu dengan algoritma Random Forest dengan akurasi 77.6% dan seiring jalannya waktu model machine learning dapat ditingkatkan dari faktor-faktor yang baru serta data yang lebih banyak.
    - Penyebab siswa drop out
        1. Kebanyakan dari mereka memiliki faktor ekonomi yang kurang memadai dan ini penyebab utama kenapa mereka keluar.
        2. Kebanyakan dari mereka menghadiri pembelajaran pada sore atau malam hari dan ini menjadi penyebab kedua.
        3. Perkerjaan orang tua juga menjadi penyebab ketiga yaitu orang tua yang tidak memiliki pendidikan yang bagus menjadikan siswa-siswa tidak melihat kesuksesan dari orang tua mereka maka mereka tidak memiliki sifat optimis.
        4. Kebanyakan dari mereka mengambil jurusan Management dan Nursing.
    - Karakteristik dari Siswa yang keluar
        1. Siswa yang kurang mampu dalam finansial.
        2. Menghadiri pembelajaran pada waktu sore atau malam.
        3. Orang tua siswa tidak memiliki jenjang pendidikan yang mumpuni.
        4. Berasal dari jurusan Managemen, Nursing dan Jurnalism and communication.
        5. Kebanyakan dari mereka belum menikah atau lanjang. 


### Rekomendasi Action Items
Berikut rekomendasi action items:
- Program Bimbingan Akademik yang Ditingkatkan
    - Institusi perlu memperkuat program bimbingan akademik dengan pendekatan yang lebih personal dan terstruktur. Dosen pembimbing akademik harus dibekali dengan akses ke dashboard monitoring mahasiswa yang memungkinkan mereka melacak perkembangan akademik secara real-time. Pertemuan bimbingan wajib harus dijadwalkan secara regular, terutama untuk mahasiswa yang teridentifikasi berisiko tinggi.
- Dukungan Finansial yang Lebih Fleksibel
    - Faktor ekonomi berperan signifikan dalam keputusan dropout, institusi perlu mengembangkan program dukungan finansial yang lebih fleksibel seperti pinjaman siswa, pembayaran berkala atau beasiswa bagi siswa yang memiliki potensi yang bagus.
- Komunikasi
    - Adanya komunikasi dapat lebih memahami apa saja kesulitan siswa yang dihadapi entah itu finansial, program belajar dan sebagainya.
- Reward
    - Adanya reward sistem akan mengurangi persentase siswa yang keluar karena reward sistem akan menumbuhkan rasa semangat mereka untuk mencapai tujuan.
- Pengukuran dan Evaluasi Program
    - Perlu adanya evaluasi program belajar siswa, seperti `apakah diakhir pembelajar A siswa mendapatkan nilai bagus?` jika tidak, evaluasi sangat membantu apa saja kekurangan program belajar tersebut.
