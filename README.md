# **📌 Panduan Penggunaan NginFaceRecog melalui Postman**

Dokumentasi ini akan membantu Anda melakukan **request ke engine Face Recognition** menggunakan **Postman**.

---

## **🛠 1. Persiapan**
Sebelum mulai, pastikan Anda memiliki:
- ✅ **Postman** ([Download di sini](https://www.postman.com/))
- ✅ **URL Server gRPC** → `172.16.13.216:8868`
- ✅ **TenantID (WAJIB)** → `5fbbbd0b-34db-43e0-8a21-a4fb1e11bae8`
- ✅ **Gambar dalam format Base64**

---

## **📌 2. Register User (Daftar Pengguna)**
gRPC Service: DriverInfranFaceID Method: RegisterUser Server: 172.16.13.216:8868

### **🔹 Body Request (JSON)**
```json
{
    "TenantID": "5fbbbd0b-34db-43e0-8a21-a4fb1e11bae8",
    "Name": "Nama Lengkap",
    "ImgData": "base64_encoded_image"
}
```
📌 Keterangan:

- TenantID (String) → Wajib, ID tenant yang telah disediakan.
- Name (String) → Nama pengguna.
- ImgData (String) → Gambar dalam format Base64.

## **📌 3. VerifyByID**
Deskripsi
Metode ini digunakan untuk memverifikasi pengguna berdasarkan UserID dan gambar (base-64).

Request
```json
{
  "UserID": "123456",
  "ImgData": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```
Langkah-langkah di Postman
- Pilih Service → YourServiceName.
- Pilih Method → VerifyByID.
- Masukkan request JSON di tab Body.
- Klik Invoke untuk mengirim permintaan.

## **📌 4. VerifyByImage**
Deskripsi
Metode ini membandingkan dua gambar untuk verifikasi.

Request
```json
{
  "ImgData1": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "ImgData2": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

Langkah-langkah di Postman
- Pilih Service → YourServiceName.
- Pilih Method → VerifyByImage.
- Masukkan request JSON di tab Body.
- Klik Invoke untuk mengirim permintaan

## **📌 5. IdentifyOne & IdentifyMany**
Deskripsi
Metode ini mencari satu/beberapa pengguna yang cocok dengan gambar yang dikirim berdasarkan TenantID.

Request
```json
{
  "TenantID": "5fbbbd0b-34db-43e0-8a21-a4fb1e11bae8",
  "ImgData": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

Langkah-langkah di Postman
- Pilih Service → YourServiceName.
- Pilih Method → IdentifyOne.
- Masukkan request JSON di tab Body.
- Klik Invoke untuk mengirim permintaan.

## **📌 Tips**
Pastikan endpoint gRPC yang digunakan sudah benar.
Gunakan gambar dalam format base64 yang valid.
Jika terdapat error, cek kembali format request atau pastikan server berjalan dengan benar.
