# AlphaGroup_JC_DS_OL_08_FinalProject
**Final Project by Alpha Team**
<hr>   

**Alan Dodi Amdani**
<br>
**Andi Setianto**
<br>
**Jessica Seanjaya**

# **DC Residential Properties**
<hr>

Link Colab: https://colab.research.google.com/drive/1duJVt91P6MUn2U0MFyl36bTqi-JTWBkM?usp=sharing <br>
Link Tableu: https://public.tableau.com/app/profile/alpha.pwdk/viz/TableuDCProperties/DashboardDCResidential <br>
Link download material: https://drive.google.com/drive/folders/1ZYBtm7hUT7WIDebW0iXiPDW4-Jk_k4aM?usp=share_link <br?
Link demo / deployment: https://dcproperties.herokuapp.com

# **Contents**
<hr>

1. Business Problem Understanding
2. Data Understanding
3. Data Cleaning & Preprocessing 
4. Exploratory Data Analysis (EDA)
5. Analytical Approach (Modelling)
6. Deployment
7. Conclusion and Recommendation
8. Data Visualization

#**1. Business Problem Understanding**
<hr>

Washington, D.C. merupakan ibu kota dari Amerika Serikat dimana menduduki peringkat keenam wilayah metropolitan terbesar di US yang paling berpendidikan dan makmur [(Wikipedia)](https://en.wikipedia.org/wiki/Washington_metropolitan_area). Washington, D.C. secara konsisten merupakan salah satu pasar paling menjanjikan dan juga menempati peringkat tinggi dalam rata-rata harga sewa  di Amerika Serikat.

Berdasarkan US Bureau of the Census estimasi populasi DC per bulan 1 Juli 2018 adalah 702,455 dan meningkat sebanyak 6,764 dari tahun 2017 [(cfo.dc)](https://cfo.dc.gov/sites/default/files/dc/sites/ocfo/publication/attachments/Economic%20and%20Revenue%20Trends%20Report_December%202018.pdf). Berdasarkan sensus data tahun 2017 sampai 2021 berdasarkan [DC DAV](https://opdatahub.dc.gov/) data unit tempat tinggal sebanyak 310,104 unit. Sebagai pusat pemerintahan nasional, biaya hidup di kota ini cukup tinggi, termasuk harga properti yang mahal. Pada tahun 2017, rata-rata harga rumah tunggal di kota ini sekitar 647.000, sedangkan rata-rata harga rumah di seluruh Amerika Serikat pada waktu yang sama adalah sekitar 399.700 menurut [Economics Research](https://fred.stlouisfed.org/series/ASPUS).

Dikutip dari [Kimberly Amadeo](https://www.thebalancemoney.com/how-does-real-estate-affect-the-u-s-economy-3306018), pada tahun 2018 konstruksi properti memberikan kontribusi sebesar USD 1,15 triliun untuk output ekonomi negara. Angka tersebut setara dengan 6,2% dari produk domestik bruto Amerika Serikat. Meskipun jumlah tersebut lebih tinggi dibandingkan dengan USD 1,13 triliun pada tahun 2017, namun tetap lebih rendah dibandingkan puncak pada tahun 2006 yang mencapai USD 1,19 triliun. Pada saat itu, konstruksi properti merupakan komponen GDP yang cukup besar sebesar 8,9%. Dengan pengaruh yang signifikan tersebut pemerintah Amerika serikat perlu memberikan kebijakan terhadap berjalannya bisnis properti yang ada. Sehingga dalam menentukan harga setiap pelaku bisnis perlu mengikuti peraturan pemerintah yang sudah ditetapkan. Pernyataan tersebut dikutip dari [Chicago Both Review](https://www.chicagobooth.edu/review/should-government-intervene-housing-market).

Selain memperhatikan kebijakan pemerintah, melakukan survei terhadap properti residensial juga diperlukan untuk memperkirakan harga yang akurat. Namun dari [Lawrence Bonk](https://www.angi.com/articles/how-much-does-land-survey-cost.htm), menurut Department of Consumer and Regulatory Affairs Washington, D.C., survei konvensional dapat menjadi sangat mahal dan membutuhkan waktu yang lama. Selain itu, metode konvensional seperti survei juga berisiko terjadi kesalahan dalam menetapkan harga, seperti overpriced dan underpriced.

## **1.1 Problem Statement**

Memperkirakan harga properti residential melalui survei bisa memakan waktu dan biaya yang cukup besar. Namun, subjektivitas penentuan harga oleh surveyor dapat menyebabkan properti dijual dengan harga yang terlalu tinggi atau terlalu rendah. Dampak dari harga yang terlalu tinggi atau rendah bisa merugikan perusahaan ataupun pembeli, seperti properti sulit terjual karena harga yang tidak bersaing pada kasus harga terlalu tinggi atau profit yang berkurang pada kasus harga terlalu rendah.

Kita dapat membagi suatu kasus menjadi beberapa komponen yaitu : Problem, data, ML Objective, Action, serta value atau hasil 
akhir yang ingin dicapai. 
1. Mengapa Machine Learning Diperlukan? 
Yaitu untuk mempermudah penentuan harga rumah
2. Data apa yang ingin diprediksi? Yaitu dataset DC Residencial Properties, untuk prediksi harga berdasarkan multiple feature.
3. Bagaimana machine 
learning dapat menyelesaikan 
masalah ?  Dengan adanya prediksi dapat membantu customer dalam membeli rumah, agar tidak terlalu mahal. Sedangkan dari sisi developer / agent, dapat menentukan harga terbaik.
4. Action apa yang 
akan kita lakukan agar objektif 
tercapai begitu kita 
mendapatkan hasil prediksi dari 
machine learning ? Melalui proses feature engineering dan selection, modelling, hyperparameter tuning, dan mendapatkan hasil prediksi.
5. Apa hasil akhir 
yang ingin dicapai? Yaitu kemudahan dalam menentukan harga, dan mengurangi biaya market riset

## **1.2 Goals**

Untuk mengatasi permasalahan di atas, tujuan dari analisa yang dilakukan adalah **memberikan prediksi harga properti residential yang lebih akurat** dan tidak terjadi overpriced dan underpriced, serta **mengurangi biaya dan waktu** yang dibutuhkan dalam survei properti.

## **1.3 Analytic Approach**

Pada Final Project ini, kita sebagai Data Science akan melakukan analisis data untuk menemukan pola dari fitur-fitur residential dari dataset DC Properties. Akan dilakukan pembuatan, evaluasi, dan implementasi model machine learning regresi sebagai tool yang dapat digunakan untuk memprediksi harga properti residential. Model machine learning ini selanjutnya akan dilakukan deployment menggunakan Flask, html dan python pada suatu website untuk memprediksi harga perumahan atau residential di Washington DC. 

Adapun keseluruhan proses analisa yang dilakukan digambarkan pada gambar di bawah ini
![M (1)](https://user-images.githubusercontent.com/116096399/228269057-81b7beab-1c38-4435-9b30-21efb7372582.png)


> Role & Interest to ML:

* Data Scientist: Membuat analisa data dan model machine learning.
* Domain Experts: Memberikan insight terkait domain knowledge untuk dasar pembuat model machine learning (contohnya ahli material, ahli geografis, dll)
* Government: Membuat kebijakan terkait penetapan harga minimal dan harga maximal jual beli property
* Customers: Menggunakan model machine learning untuk mendapatkan estimasi harga property yang akan dibeli, sesuai dengan preferensi pengguna.
* Shareholders (perusahaan property): Menggunakan product model machine learning untuk menentukan harga jual beli property.

Di atas adalah goals akhir dari pembuatan machine learning ini, yaitu agar model ini dapat digunakan oleh share holders, baik melalui mobile maupun desktop.
![download](https://user-images.githubusercontent.com/116096399/228269209-32e7909e-d4e6-4688-ac21-200292e66c2e.png)

## **1.4 Design Thinking**
Apa saja proses yang perlu kita lakukan dalam pengerjaan final project ini dengan menggunakan design thinking?

Berikut adalah proses Design Thinking Mapping untuk pembuatan machine learning regresi yang digunakan untuk memprediksi harga rumah atau residential di Washington DC
![download](https://user-images.githubusercontent.com/116096399/228269256-8818378c-e957-4629-b4c0-098efbf3706b.png)

## **1.5 Matrix Evaluation**

Metrik evaluasi yang akan digunakan adalah:

- **R-Square**

Metrik ini mewakili persentase dari semua fitur yang mempengaruhi target. Semakin tinggi nilai persentase, semakin besar pengaruh fitur terhadap target. Metrik ini hanya akan digunakan saat memilih model dasar karena dari setiap model dapat dilihat dengan jelas seberapa besar persentase pengaruh fitur dari setiap model terhadap target. Karena itu adalah persentase, nilai metrik adalah 0 hingga 1.

- **MAE (Mean Absolute Error)**

Metrik ini mewakili perbedaan rata-rata absolut antara hasil yang diprediksi dan hasil aktual. Metrik ini disarankan karena data yang digunakan adalah data yang memiliki beberapa outliers. Karena nilai absolut, metrik ini tidak peduli tentang perbedaan yang menghasilkan hasil negatif. Semakin kecil nilai metrik ini, semakin baik hasil prediksi yang didapat

- **MAPE (Mean Absolute Percentage Error)**

Metrik ini mewakili persentase perbedaan rata-rata absolut antara hasil yang diprediksi dan hasil aktual. Bedanya dengan MAE, metrik ini perlu mengubah hasil perbedaan menjadi persentase saja.

- **RMSE (Root Mean Square Error)**   
Metrik ini mewakili tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi “ observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai observasinya. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.

#**2. Data Understanding**

[Sumber Dataset](https://www.kaggle.com/datasets/christophercorrea/dc-residential-properties?select=raw_address_points.csv)

Secara umum, pemahaman data digunakan untuk memeriksa data dengan tujuan mengenali masalah pada data yang telah diperoleh. Langkah ini penting dalam analisis data karena memberikan dasar analitik untuk penelitian dengan merangkum informasi dan menemukan potensi masalah pada data.

<p>

Setiap baris dalam dataset mencakup informasi properti yang aktual di Washington, D.C. yang dikumpulkan pada bulan Juli 2018.

Attribute Information:
| Attribute | Data Type | Description | Variable Type  |
| --- | --- | --- | --- |
| Unnamed: 0 | Int | Index | - |
| BATHRM | Int | Jumlah kamar mandi | Quant. Diskrit |
| HF_BATHRM | Int | Jumlah kamar mandi tanpa *bathtub* atau *shower* | Quant. Diskrit |
| HEAT | Object | Tipe pemanas ruangan | Qual. (Nominal) |
| AC | Object | Pendingin ruangan, Y - Ada, N - Tidak ada | Qual. (Nominal) |
| NUM_UNITS | Float | Jumlah unit | Quant. Diskrit |
| ROOMS | Int | Jumlah ruangan | Quant. Diskrit |  
| BEDRM | Int | Jumlah kamar tidur | Quant. Diskrit|
| AYB | Float | Tahun properti dibangun | Quant. Kontinu |
| YR_RMDL | Float | Tahun properti direnovasi yang mengubah struktur | Quant. Kontinu |
| EYB | Int | Tahun properti direnovasi tanpa mengubah struktur| Quant. Kontinu | 
| STORIES | Float | Jumlah lantai | Quant. Diskrit |
| SALEDATE | Object | Tanggal penjualan terbaru | Quant. Kontinu |  
| PRICE | Float | Harga penjualan terbaru | Quant. Kontinu |
| SALE_NUM | Int | Jumlah penjualan sejak tahun 2014 | Quant. Diskrit |
| GBA | Float | Luas total area bangunan dalam squarefeet | Quant. Kontinu |
| BLDG_NUM | Int | Jumlah bangunan dalam properti | Quant. Diskrit |
| STYLE | Object | Style properti | Qual. (Nominal) |
| STRUCT | Object | Struktur bangunan | Qual. (Nominal) |
| GRADE | Object | Kualitas properti | Qual. (Ordinal) |
| CNDTN | Object | Kondisi properti | Qual. (Ordinal) |
| EXTWALL | Object | Jenis dinding eksterior | Qual. (Nominal) |
| ROOF | Object | Jenis atap rumah | Qual. (Nominal) |  
| INTWALL | Object | Jenis dinding interior | Qual. (Nominal) |
| KITCHENS | Float | Jumlah dapur | Quant. Diskrit |
| FIREPLACES | Int | Jumlah perapian | Quant. Diskrit |
| USECODE | Int | Kode penggunaan properti | Qual. (Nominal) | 
| LANDAREA | Int | Luas tanah properti dalam square feet | Quant. Kontinu |
| GIS_LAST_MOD_DTTM | Object | Tanggal data diperbaharui | Quant. Kontinu |  
| CMPLX_NUM | Float | Nomor komplek | Qual. (Nominal) |
| LIVING_GBA | Float | Luas area tempat tinggal dalam square feet | Quant. Kontinu |
| FULLADDRESS | Object | Alamat jalan | Qual. (Nominal) |
| CITY | Object | Kota | Qual. (Nominal) |
| STATE | Object | Negara bagian | Qual. (Nominal) |
| ZIPCODE | Float | Kode pos | Qual. (Nominal) |
| NATIONALGRID | Object | Informasi terkait power station terdekat | Qual. (Nominal) | 
| LATITUDE | Float | Latitude lokasi properti | Qual. (Nominal) |
| LONGITUDE | Float | Longitude lokasi properti | Qual. (Nominal) |  
| ASSESSMENT_NBHD | Object | ID lingkungan | Qual. (Nominal) |
| ASSESSMENT_SUBNBHD | Object | ID sub lingkungan | Qual. (Nominal) |
| CENSUS_TRACT | Float | Region sensus | Qual. (Nominal) |
| CENSUS_BLOCK | Object | Blok pengambilan sensus | Qual. (Nominal) |
| WARD | Object | Distrik | Qual. (Nominal) |
| SQUARE | Object | Square (SSL - (Square, Suffix, Lot)) | Qual. (Nominal) |
| X | Float | Longitude lokasi properti | Qual. (Nominal) |
| Y | Float | Latitude lokasi properti | Qual. (Nominal) |
| QUADRANT | Object | Kuadran kota (NE,SE,SW,NW) | Qual. (Nominal) |

Property Overview:
| No | House Features    |                                     | Construction Details |                             |
|----|-------------------|-------------------------------------|----------------------|-----------------------------|
|    | Based on:         | Consist of:                         | Based on:            | Consist of:                 |
| 1  | Interior Details  | -        Bedrooms                   | Property             | -        Story              |
|    |                   | -        Bathrooms                  |                      | -        Style              |
|    |                   | -        Half Bathrooms             |                      |                             |
| 2  | Heating           | -        Forced Air                 | Wall Type            | -        Exterior           |
|    |                   | -        Electric Based             |                      | -        Interior           |
|    |                   | -        etc                        |                      |                             |
| 3  | Cooling           | -          Air Conditioner          | Roof Type            | -        Slate              |
|    |                   |                                     |                      | -          Concrete         |
| 4  | Interior Features | -        Kitchens                   | Condition            | -        Property Condition |
|    |                   | -        Fireplaces                 |                      | -        Property Grade     |
| 5  | Other Features    | -        Gross Building Area        | Notable Dates        | -        Year Built         |
|    |                   | -        Living Gross Building Area |                      | -        Year Remodel       |

#**3. Data Preprocessing**

Pada tahap ini, data akan dibersihkan sehingga dapat digunakan untuk analisis selanjutnya. Beberapa hal yang perlu dilakukan selama proses pembersihan data adalah:

- Pengecekan duplikasi data.
- Menentukan dan menangani data anomali.
- Menghapus fitur yang tidak relevant untuk masalah yang sedang dianalisis.
- Mengatasi missing value, baik dengan menghapus fitur yang tidak penting atau mengisi dengan nilai yang paling masuk akal.

Dalam menganalisis data, kualitas data akan menentukan hasil akhir. Tak peduli sebaik apapun hasil analisis, jika kualitas data rendah, hasilnya akan terdistorsi atau tidak memuaskan. Penentuan apakah kualitas data baik atau tidak ditentukan pada proses ini.

#**4. Data Analysis**

Pada proses data analysis, kita akan melakukan proses dimana data diolah dengan tujuan untuk menemukan informasi yang bermanfaat dan dapat digunakan sebagai dasar dalam pengambilan keputusan untuk mengatasi suatu masalah. Setelah data melalui proses pembersihan, selanjutnya data tersebut disajikan dengan cara yang menarik dan mudah dipahami oleh orang lain, umumnya dalam bentuk grafik atau plot.

#**5. Modelling**

Sebelum kita membahas lebih lanjut, pertama kita akan membahas apa itu regresi terlebih dahulu. Model regresi sering digunakan dalam penelitian kuantitatif. Regresi dilakukan dalam pengujian pengaruh, biasanya menguji antara pengaruh variable independen terhadap variable dependen (Sekaran & Bougie, 2016). 

Kemudian kita membahas bahwa terdapat regresi linear sederhana dan regresi linear berganda. Analisis regresi sederhana adalah sebuah metode pendekatan untuk pemodelan hubungan antara satu variabel dependen dan satu variabel independen. Dalam model regresi, variabel independen menerangkan variabel dependennya. Dalam analisis regresi sederhana, hubungan antara variabel bersifat linier, di mana perubahan pada variabel X akan diikuti oleh perubahan pada variabel Y secara tetap. Sedangkan regresi linear berganda adalah apabila variable independennya lebih dari satu, dalam artian dua, tiga, dan seterusnya.

Dalam kasus ini, kita akan coba menerapkan **multiple linear regression**

#**6. BONUS! Deploy Model to Web Base**

Deployment Machine Learning adalah proses membuat model machine learning yang telah dilatih (trained) dapat digunakan dalam aplikasi yang dapat diakses oleh pengguna akhir secara online.

Deployment machine learning memiliki peran penting dalam pemanfaatan model machine learning secara optimal dalam berbagai aplikasi yang dapat mempermudah dan mempercepat pengambilan keputusan untuk pembelian dan penjualan properti.
![Screenshot 2023-03-28 at 19 58 07](https://user-images.githubusercontent.com/118766459/228267573-c39f8e54-8ea5-43d9-8a46-577f054228d8.png)
![Screenshot 2023-03-28 at 19 59 13](https://user-images.githubusercontent.com/118766459/228267597-9c9750d0-4098-4542-a2e2-de6b38107f3c.png)
![Screenshot 2023-03-28 at 19 59 21](https://user-images.githubusercontent.com/118766459/228267611-2e25132b-ad36-4567-84b5-cb3954aa71c3.png)
<br>
Link video penggunaan aplikasi dapat diakses melalui https://drive.google.com/drive/folders/1ZYBtm7hUT7WIDebW0iXiPDW4-Jk_k4aM?usp=share_link



#**7. Conclusion and Recommendation**
###**7.1 Conclusion**
**Berikut resume dari performa pemodelan yang telah dilakukan**

<br>

Untuk pengecekan awal kita menggunakan 9 algoritma yaitu :
* LinearRegression
* KNeighborsRegressor
* DecisionTreeRegressor
* RandomForestRegressor
* GradientBoostingRegressor
* Lasso
* Ridge
* AdaBoostRegressor
* XGBRegressor

<br>

Dan hasilnya sebagai berikut :

Kami memahami bahwa sebagai data scientist perlu mencoba untuk memaksimalkan performa model sampai yang paling baik. Oleh karenanya kami mencoba melakukan beberapa kali iterasi seperti yang sudah dilakukan pada proses modeling di atas dengan feature engineering menggunakan pertimbangan correlation dan multikolinearity. Namun hasil dari iterasi tersebut ternyata **tidak memberikan peningkatan performa** sehingga kita kembali ke base model dengan algoritma yang paling maksimal nilainya. Dapat dilihat algoritma model yang memberikan error terkecil dan R-square yang tertinggi adalah **XGBRegressor** disusul oleh **Random Forest Regresor**.

<br>

Percobaan lainnya untuk memaksimalkan performa model adalah dengan melakukan hyper parameter tuning. dari proses yang sudah dilakukan didapat hasil sebagai berikut:

<br>

| **Model** | **MAE Before Tuning** | **MAE After Tuning**|
| --- | --- | --- |
| **XGBRegressor** | 76398.224997 | 73699.792262	 |
| **RandomForestRegressor** | 79824.261326 |97358.423595 |

Dari Tuning yang dilakukan terlihat bahwa algoritma model yang memberikan performa terbaik dengan memberikan nilai eror paling kecil setelah di tuning adalah **XGBRegressor**.

Parameter terbaik yang diberikan dari hasil tuning untuk XGBRegressor adalah:

1. model__subsample : 0.5,
2. model__n_estimators : 182, 
3. model__max_features: 0.8, 
4. model__max_depth: 9, 
5. model__learning_rate': 0.15

Maka dari itu dapat kita simpulkan menjadi beberapa point penting:

1. Dataset ini cenderung lebih optimal dengan algoritma yang tree based serta boosting. Jika kita lihat berdasarkan MAE-nya, 2 performa terbaik ada pada Random Forest dan Gradient Boosting.
Hal ini bisa disebabkan oleh beberapa faktor, dari yang saya baca dari https://www.summitllc.us/blog/advantages-of-tree-based-modeling , tree-based model ini cocok untuk dataset dengan berbagai tipe data, kita tahu bahwa dataset DC Residential Properties terdapat beberapa tipe data, yaitu numerikal dan kategorikal. 
Kemudian tree based model juga cocok untuk data yang sifatnya tidak terdistribusi normal. Selain itu tree based model juga dapat diterapkan pada data yang mengalami masalah non linieritas.

2. Jika dilihat dari nilai RMSE yang dihasilkan oleh model setelah dilakukan hyperparameter tuning, yaitu sebesar 131262.23 USD, kita dapat menyimpulkan bahwa bila nanti model yang kita buat ini digunakan untuk memperkirakan harga rumah pada rentang nilai seperti yang di-training terhadap model, maka perkiraan harganya rata-rata akan meleset kurang lebih sebesar 131262.23 USD dari harga yang mungkin seharusnya. Tetapi, tidak menutup kemungkinan juga prediksinya meleset lebih jauh karena bias yang dihasilkan model masih cukup tinggi pada harga prediksi yang semakin tinggi, seperti visualiasi yang telah ditunjukkan pada asumsi asumsi sebelumnya. 
Bisa dibuktikan juga pada pengujian model terhadap data test memiliki nilai yang meleset yang paling tinggi diatas 2021013 USD 
 (difference). Hal ini terjadi karna adanya bias yang cukup tinggi pada model. 

3. Selain itu, jika kita lihat berdasarkan nilai MAPE, untuk hasil MAPE  0.147157 x 100 = 14.7157% / di bawah 20% dapat tergolong baik

4. Hasil dari Rsquare yang di atas 92% sudah cukup baik, artinya bahwa 92% variabilitas yang diamati dalam variabel target dapat dijelaskan oleh model regresi.
Ingat, bahwa R Square berfungsi untuk mengukur seberapa baik regresi merepresentasikan data.

###**7.2 Manfaat Bisnis**
1. Berdasarkan informasi yang saya baca dari https://hingemarketing.com/blog/story/cost_and_benefits_of_market_research , dengan Traditional Valuation, market researchers butuh waktu 2 - 8 minggu untuk melakukan riset harga terbaik.
Sedangkan dengan Advance Valuation (machine learning model), kita dapat melakukan prediksi 1 unit residence price kurang dari satu hari.
Anggaplah biaya market riset dalam 4 minggu adalah 100 USD per hari, jika dalam 4 minggu(dikurangi weekend / hari istirahat) maka total biayanya adalah 2000 USD. Jika dibandingkan dengan menggunakan ML yang hanya kurang dari 1 hari, maka perbandingannya adalah 100 USD : 2000 USD, atau lebih hemat 95 % (di luar estimasi biaya pengembangan ML). 
Apa artinya? secara tidak langsung kita bisa menurunkan biaya riset market, atau mengalihkannya untuk pengembangan ML. Tentunya akan berdampak juga untuk strategi marketing dari agen properti, karena bisa lebih cepat dalam menentukan harga, dan bisa menerapkannya pada berbagai bentuk, misalnya untuk bekerja sama dengan aplikasi jual beli rumah.
Selain itu dari sisi Buyer, dapat mereduksi resiko pembelian residence dalam harga yang terlalu mahal
Dari sisi Penjual, akan bermanfaat dalam menentukan harga terbaik, supaya tidak terlalu mahal ataupun terlalu murah (sweet spot).

2. Meningkatkan Daya Saing Bisnis: Dengan memanfaatkan machine learning price prediction, bisnis property dapat menjadi lebih kompetitif dalam pasar yang semakin sengit. Hal ini karena bisnis property dapat menawarkan harga yang lebih tepat, penawaran yang lebih personal dan relevan, serta meningkatkan efisiensi operasi bisnis.
3. Pengambilan Keputusan yang Lebih Baik: Dengan memiliki prediksi harga yang akurat, bisnis property dapat membuat keputusan yang lebih baik dalam menentukan harga jual, menilai nilai investasi, dan melakukan investasi yang lebih cerdas. Hal ini dapat membantu bisnis property untuk mengoptimalkan keuntungan dan mengurangi risiko kehilangan investasi.

**7.3 Limitasi**
1. Karena adanya penghapusan terhadap outlier, maka akan ada limit pada batas maksimal harga dan size yang dimasukan, untuk harga maximal yang bisa dimasukan adalah 5685000.0 USD, sedangkan size minimal yang bisa dimasukan adalah 58304

2. Pada fitur ageBuilding tidak dinamis / perlu diupdate kembali setiap ada pertambahan tahun. 

3. Karena kita menggunakan tree-based model, akan ada beberapa kekurangan,   

 - pertama, jika datanya terlalu besar (big data), maka akan terlalu banyak node yang terbuat, sehingga model menjadi terlalu kompleks dan overfitting. 

 - Selain itu terkait reusability-nya, karena perubahan kecil pada data (misal suatu saat ingin memasukan dataset yang serupa) dapat menyebabkan perubahan struktur secara signifikan (tree tidak stabil)

4. Masih sedikit bias seperti yang sudah dijelaskan pada Jika kita lihat dari grafik actual vs prediction, prediksi sudah merapat pada satu titik, namun ada sedikit kecenderungan untuk menyebar, terutama pada nilai di atas 2000000 USD. Hal ini menandakan masih adanya potensi bias pada prediksi yang sudah dibuat, maka perlu menjadi concern ketika harga di atas 2000000 USD.

###**7.4 Rekomendasi dan Pengembangan**
####**7.4.1 Rekomendasi Terhadap Dataset & Model**
**Untuk memperbaiki hasil prediksi di masa depan, beberapa hal yang dapat dilakukan terhadap dataset dan model adalah:**

<br>

**Rekomendasi untuk Dataset**

<br>

1. Menambahkan sumber data agar model memiliki lebih banyak informasi untuk dipelajari, yaitu dengan menambahkan data teraktual untuk perumahan atau residential yang ada di washington DC.
2. Menambahkan fitur seperti jarak ke sekolah, transportasi umum, pusat perbelanjaan bisa dilakukan sebagai salah satu cara untuk menentukan apakah properti tersebut terletak di lokasi yang strategis atau tidak. Dengan cara ini, dapat memberikan penjelasan tambahan mengenai lokasi properti dan seberapa dekat properti tersebut dengan fasilitas-fasilitas yang penting dalam kehidupan sehari-hari.
3. Dalam pengumpulan dataset, diharapkan genap hingga 1 tahun sehingga profil penjualan setiap tahunnya dan setiap bulannya dapat dibandingkan secara apple to apple karena jumlah data yang sama.
4. Untuk meningkatkan akurasi model prediksi, salah satu cara yang dapat dilakukan adalah dengan menambahkan lebih banyak data, terutama pada kisaran harga di antara 2 juta USD - 6 juta USD. Dengan cara ini, model akan memiliki lebih banyak data untuk dipelajari dan dapat menghasilkan prediksi yang lebih akurat pada properti dengan harga yang lebih tinggi.
5. Sudah sangat wajar semakin tinggi harga suatu rumah, kepemilikan unit perumahan dengan harga tinggi tentunya semakin sedikit jumlahnya. Oleh karena itu perlu dipertimbangkan untuk mulai membagi dan mengelompokan perumahan mana saja rumah yang tergolong mewah dengan rumah yang biasa di Washington DC. Hal ini sangat membantu terhadap performa model yang akan dibuat karena distribusi data harga yang ada akan lebih normal.
6. Untuk memastikan kualitas data, penting untuk memeriksa dan menghindari kesalahan input data pada saat pengumpulan dataset. Kesalahan input seperti kesalahan dalam tahun, nomor atau fitur lainnya dapat mempengaruhi kualitas data dan menghasilkan kesimpulan atau prediksi yang tidak akurat. Oleh karena itu, diperlukan proses verifikasi data secara hati-hati dan sistematis untuk memastikan integritas dan konsistensi data. Selain itu juga bisa ditambahkan dictionary atau drop down untuk setiap option value pada fitur terkait, agar tidak terjadi kesalahan input

<br>

**Rekomendasi untuk Model**

<br>

1. Di masa depan, model ini memerlukan dukungan dari model klasifikasi yang dapat menentukan apakah harga penjualan/pembelian rumah tersebut "menguntungkan atau tidak" berdasarkan harga rumah yang telah terjual/dibeli sebagai variabel target. Selain itu, penerapan model unsupervised juga bisa digunakan seperti K-means clustering (model ini digunakan untuk mengelompokkan data ke dalam kelas yang berbeda, dengan meminimalkan jarak antara setiap data dengan pusat kelompok) atau Principal Component Analysis / PCA (model ini digunakan untuk mengekstraksi fitur penting dari data, dengan mengurangi dimensi data ke fitur-fitur yang paling signifikan dan menghilangkan fitur yang tidak relevan).
2. Mengingat distribusi data dari rumah rumah yang umum dan rumah mewah bisa dipastikan akan berbeda secara signifikan. Ada pilihan untuk membuat model terpisah untuk rumah mewah atau memfokuskan pada penjualan rumah biasa untuk kebutuhan sehari-hari, karena keberadaan data rumah mewah dapat memberikan dampak besar pada model.
3. Karena adanya kelompok data condominium dalam dataset, di masa depan bisa dibuat sebuah model untuk properti Condominium yang dapat digunakan untuk melengkapi prediksi harga semua jenis properti di Washington DC.
4. Model ini tentu masih dapat diimporvisasi agar dapat menghasilkan prediksi yang lebih baik lagi. Namun, kita dapat melakukan A/B testing terhadap model yang sudah dibuat pada project ini untuk mengetahui tingkat efektifitas penggunaan model.

5. Mengecek prediksi mana saja yang memiliki nilai error yang tinggi. Kita dapat mengelompokkan error tersebut ke dalam grup overestimation dan underestimation, lalu memilih 5% error paling ekstrim saja untuk tiap grup. Nantinya pengelompokkan akan menjadi 3 grup, yaitu overestimation (5%), underestimation (5%), dan grup mayoritas yang error-nya mendekati nilai mean (90%). Setelahnya kita bisa mengecek hubungan antara error tersebut dengan tiap variabel independen. Pada akhirnya kita dapat mengetahui sebenarnya variabel mana saja dan aspek apa yang menyebabkan model menghasilkan error yang tinggi, sehingga kita bisa melakukan training ulang dengan penerapan feature engineering lainnya.

6. Pergunakan Grid Search, karena biasanya Grid Search juga dapat memeliki dampak pada performa secara lebih baik. Namun tentu ada trade off-nya, yaitu computational cost-nya, berarti membutuhkan device dengan spesifikasi yang baik.

7. Ada sebuah topik menarik, yaitu terkait House Price Bubble, terjadi ketika suatu harga properti naik secara signifikan dan kemudian harganya melambung turun drastis. Hal seperti ini bisa terjadi ketika harga properti terus naik tinggi sementara kapabilitas orang untuk membeli properti menurun (demand turun), atau mereka tidak ada kekuatan membeli (purchasing power). Di samping itu, ada sebuah pernyataan dari **Roberts, Lawrence (2008)**,
agar pasar otomatis bisa menyadari adanya price bubble
disarankan perubahan cara penilaian yang sudah ada yang 
menggunakan teknik sales comparison approach menjadi teknik income 
approach. Perubahan cara penilaian ini disebut oleh Roberts (2008) 
dengan sebutan market solution. Sales comparison approach didasarkan 
pada harga bangunan yang dianggap relative sama pada transaksi jual 
beli terakhir sehingga kenaikan harga suatu rumah memicu kenaikan 
harga rumah lainnya. Kemudian kita juga bisa mempertimbangkan income / penghasilan rata rata pada tiap daerah. Meskipun secara keuntungan berbisnis hal ini perlu diteliti lebih lanjut.

####**7.4.2 Rekomendasi Terhadap Bisnis**
1. Dengan adanya machine learning, maka proses riset terhadap harga tentu menjadi lebih cepat, maka kita bisa bekerja sama dengan aplikasi jual beli rumah. Misalnya dengan memberikan rekomendasi harga pada pemilik rumah yang ingin menjual unit-nya. Contoh proses bisnisnya secara singkat, seller input data rumah/ spesifikasi rumah, kemudian aplikasi akan memunculkan rekomendasi harga jual. Sehingga seller pun tidak bingung dalam menentukan harga, tidak terlalu mahal dan tidak terlalu murah. Ke depan bisa dievaluasi, apakah dengan menerapkan model / rekomendasi tersebut traffic penjualannya semakin bagus? apakah akan lebih cepat terjual? Jika iya, berarti model tersebut berguna bagi bisnis. Maka hubungan antara Model Developer dan Aplikasi Jual Beli Properti bisa menjadi saling menguntungkan (misalnya dalam bentuk sharing profit)
2. **Mulai dari Kasus Penggunaan yang Sederhana** Bisnis sebaiknya memulai dengan kasus penggunaan yang sederhana dan kemudian memperluas penggunaan teknologi machine learning sesuai dengan kebutuhan bisnis. Hal ini akan membantu dalam memahami cara kerja teknologi machine learning dan meminimalkan risiko kesalahan dalam penggunaan teknologi tersebut. Seperti yang kita ketahui bersama kesalahan kesalahan yang terjadi bisa memnyebabkan tambahan cost pada perusahaan.
3. **Libatkan Tim Ahli** Jika bisnis tidak memiliki tim ahli dalam machine learning, maka sebaiknya bisnis melibatkan tim ahli untuk membantu dalam penggunaan teknologi machine learning. Tim ahli dapat membantu bisnis dalam memilih algoritma regresi yang tepat, mempersiapkan data, dan melatih model regresi atau model model machine learning lainnya.
4. **Lakukan pengontrolan atau pengendalian** yang meliputi pengujian dan evaluasi sistem, implementasi, dan pengelolaan sistem yang dihasilkan. Dalam tahap ini, perusahaan perlu melakukan uji coba terhadap sistem machine learning yang telah dibangun dan melakukan evaluasi terhadap kinerja sistem tersebut. Perusahaan juga perlu melakukan pemeliharaan dan pengelolaan sistem secara berkala untuk memastikan sistem dapat berjalan dengan baik.
5. **Lakukan update dataset modelling secara berkala** adakalanya ketika trend/jaman yang berubah, suatu model tidak memberikan hasil yang valid kembali. Oleh karena itu perlu dilakukan update dataset modelling sehingga model yang dikembangkan dapat mengikuti perkembangan trend/jaman.
6. Tekait kebijakan Qualified dan Unqualified perlu dikaji kembali relevansinya, disarankan menggunakan segementasi ataupun teknik clustering, karena sudah dijabarkan pada saat proses data cleaning, bahwa kolom QUALIFIED pada data properti menunjukkan apakah harga properti tersebut sesuai dengan harga pasar yang wajar, berdasarkan penilaian dari pemerintah. Jika data unqualified digunakan dalam pemodelan, maka hasil analisis tidak akurat dan tidak sesuai dengan tujuan analisis, yaitu untuk memprediksi harga sesuai dengan nilai pasar yang wajar. Hal ini dapat menyebabkan kemungkinan adanya underpricing atau overpricing.

**8. Data Visualization**
 <br>
![Dashboard DC Residential](https://user-images.githubusercontent.com/118766459/228267870-080016bb-0029-4962-a0a2-6be9385aca05.png)

