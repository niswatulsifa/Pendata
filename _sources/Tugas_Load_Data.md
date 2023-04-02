### Load Data 

#### Cara Load data file CSV ke Excel :

1. Downoad file CSV yang akan di load terlebih dahulu.
2. Berhubung Microsoft Excel yang ada di laptop saya itu versi 2013 jadi untuk load data langkah pertama yang dilakukan yaitu mengklik Data terlebih dahulu.
3. Kemudian di Data, klik bagian Get Data From Text.
4. Lalu akan diarahkan ke File Explorer untuk memilih file CSV yang telah di download sebelumnya, kemudian klik import.
5. Setelah klik import akan menampilkan jendela Import Wizard dan akan ada pilihan antara Delimited atau Fixed Width, nah disitu pilih yang Delimited. Jangan lupa centang juga “My data has headers” lalu klik next.
6. Di jendela Import Wizard selanjutnya akan ada pilihan Tab, Semicolon, Comma, Space, Other. Di jendela tersebut klik yang bagian Comma kemudian Next dan Finish.
7. File CSV sudah di Load di Excel.

#### Cara Load data file CSV ke Power BI :

1. Downoad file CSV yang akan di load terlebih dahulu.
2. Download juga aplikasi Power BI kemudian jalankan.
3. Lalu masuk ke Power BI tersebut dan pilih Get Data, pilih bagian Text/CSV lalu klik Connect.
4. Kemudian akan di arahkan ke File Explorer untuk memilih file CSV yang sudah di download tadi.
5. Setelah itu klik Load dan centang semua kolom pada Data di file Iris sehingga muncul dalam bentuk diagram.

#### Cara Load data postgreSQL ke Cloud Server :

1. Membuat akun elephantsql terlebih dahulu di website elephantsql.com dengan klik login, kemudian pilih Create New instance. Kemudian nantinya akan muncul Instance yang sudah dibuat.
2. Setelah itu buka PgAdmin dan hubungkan dengan cara buat server baru dengan memasukkan Hostname, Username, dan Password sesuai dengan yang sudah dibuat di akun elephant tadi.
3. Cari database di PgAdmin sesuai dengan elephant yang telah dibuat.
4. Kemudian Import file CSV tadi ke dalam database menggunakan Query Tool.
5. Lalu buka Power BI, kemudian klik menu home dan klik get data lalu klik more. 
6. Setelah itu pilih Postgresql database kemudian klik Connect.
7. Lalu masukkan nama Server dan Database sesuai akun elephant yang sudah dibuat kemudian klik Ok.
8. Masukkan Username dan Password sesuai dengan akun elephant yang sudah dibuat, lalu klik Connect. Maka akan muncul data yang ada di Database pada akun elephant kemudian pilih public.iris dan klik Load.
9. Setelah itu klik bagian file public iris kemudian centang semua kolom dan akan muncul data Iris.csv dalam bentuk diagram. Lalu save data atau file yang sudah di load tadi.

#### Cara Load data dari postgreSql Local :

1. Sebelum itu pastikan sudah menginstall PgAdmin dan juga Power BI, pastikan juga sudah membuat database baru di PgAdmin kemudian file CSV yang akan di import sudah ada di dalam database tersebut.
2. Setelah itu masuk pada Power BI kemudian klik Get Data.
3. Kemudian pilih database dengan nama postgreSql database lalu klik Connect.
4. Isi server dan database sesuai dengan server dan database yang ada pada PgAdmin Local dan kemudian klik Ok.
5. Pilih file CSV yang akan ditampilkan kemudian centang file tersebut dan klik Load.
6. Setelah itu klik bagian file yang sudah di Load tadi, centang semua kolom dan data akan tampil dalam bentuk diagram. Lalu save data yang sudah di Load tadi.

#### Cara Load data dari MySQL database ke Power BI :

1. Jalankan MySQL terlebih dahulu, pastikan di dalam database MySQL sudah ada file CSV bisa dengan cara import file atau dengan Kueri.
2. Setelah itu masuk pada Power BI kemudian klik Get Data.
3. Kemudian pilih database dengan nama MySQLdatabase lalu klik Connect.
4. Isi server dan database sesuai dengan server dan database yang ada pada MySQL anda kemudian klik Ok.
5. Pilih file yang ada dalam MySQL yang akan ditampilkan kemudian centang file tersebut dan klik Load.
6. Setelah itu klik bagian file yang sudah di Load tadi, centang semua kolom dan data akan tampil dalam bentuk diagram. Lalu save data yang sudah di Load tadi.