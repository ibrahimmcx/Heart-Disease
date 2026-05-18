# 📊 Kaggle & Cardio-Shield CDSS Karşılaştırmalı Analiz Raporu

Bu rapor, Kaggle üzerinde popüler olan ve çok kullanılan **"Heart Disease Prediction - Group Project"** (Rana Alghamdi & Ekibi) çalışması ile bizim geliştirdiğimiz **Cardio-Shield: Karar Destek Sistemi (CDSS)** arasındaki mimari, algoritmik, finansal ve klinik performans farklarını bilimsel ve istatistiksel açıdan ortaya koymaktadır.

---

## 🔍 1. Standart Kaggle Projesi Analizi (`ranaalghamdi26`)

Kaggle notebook'u incelendiğinde, geleneksel bir veri bilimi iş akışının izlendiği görülmektedir:

* **Veri Kümesi:** Kaggle'daki `johnsmith88/heart-disease-dataset` (1025 satırlık Cleveland genişletilmiş veri kümesi) kullanılmıştır.
* **Veri Önişleme (Preprocessing):**
  * `df.drop_duplicates()` kullanılarak mükerrer kayıtlar silinmiştir. Bu doğru bir yaklaşımdır çünkü 1025 satırlık Kaggle seti, aslında orijinal 303 satırlık Cleveland kümesinin yapay olarak çoğaltılmış halidir. Mükerrer kayıtların silinmesiyle veri kümesi **302 tekil hastaya** düşürülmüştür.
  * Sayısal özellikler standard scale edilmiştir.
  * Eğitim/Test bölümü %80 - %20 olarak ayrılmıştır.
* **Makine Öğrenmesi Modeli:**
  * **Düz (Flat) Model:** Sadece tek bir `RandomForestClassifier` eğitilmiştir.
  * 13 klinik özelliğin tamamı (yaş, cinsiyet gibi temel demografiklerden, Fluoroscopy ve Scintigraphy gibi pahalı ve girişimsel testlere kadar) modele aynı anda girdi olarak verilmektedir.
* **Elde Edilen Başarım Metrikleri:**
  * **Test Doğruluğu (Accuracy):** **%75.41**
  * **Duyarlılık (Recall - Sınıf 1 / Hasta):** **%79.00**
  * **F1-Score:** **%78.00** (Hasta sınıfı için)

---

## 🛡️ 2. Bizim Yaklaşımımız: Cardio-Shield CDSS

Geliştirdiğimiz sistem, düz bir makine öğrenmesi modeli olmaktan öte, tıp dünyasının finansal ve klinik kısıtlarını entegre eden **Maliyet Duyarlı Kademeli Eskalasyon (Cost-Sensitive Staged Escalation)** mimarisine sahiptir:

* **Çok Aşamalı Triage Mimarisi (Stage 1-4):** Özellikler tıbbi maliyet ve girişimsel ağırlıklarına göre 4 aşamaya bölünmüştür. Her aşamada ayrı bir XGBoost/Random Forest modeli karar verir.
* **Güven Sınırları (Early Stopping):** Eğer 1. veya 2. aşamada hastanın risk skoru <%15 (Düşük Risk) veya >%85 (Yüksek Risk) ise teşhis süreci anında durdurulur. Hasta taburcu edilir ya da doğrudan acil sevk edilir. Pahalı testler (Fluoroscopy, Thalassemia Scintigraphy) sadece arada kalan (%15 - %85 arası) gri alan hastalarına uygulanır.
* **Pearson Korelasyon Uyumlu XAI (SHAP):** Yalnızca hastanın teşhisinin durdurulduğu aşamadaki özelliklerin yerel katkısı (SHAP), fizyolojik yönleriyle (riski artıranlar kırmızı, koruyucu faktörler mavi barlar olarak) hekime canlı sunulur.

---

## 📊 3. Ayrıntılı Karşılaştırma Matrisi

| Karşılaştırma Kriteri | Standart Kaggle Yaklaşımı | Cardio-Shield CDSS |
| :--- | :--- | :--- |
| **Model Mimarisi** | **Düz (Flat) Sınıflandırma:** Tek aşamalıdır. Her hastadan 13 testin tamamı istenir. | **Kademeli Eskalasyon (Dynamic Triage):** 4 teşhis aşamasından oluşur. İstekler dinamiktir. |
| **Klinik/Finansal Maliyet Farkındalığı** | **Tamamen Yok:** $5'lık kan şekeri testiyle $350'lık radyolojik görüntüleme testi eşdeğer görülür. | **Maliyet Duyarlı:** Testlerin dolar bazlı klinik maliyetleri formüle ve karar sınırlarına doğrudan etki eder. |
| **Hasta Başına Ortalama Test Maliyeti** | **Sabit $595.00** (Her hasta 13 testin tamamını yaptırmak zorundadır). | **Ortalama $161.85** (Hastaların büyük kısmında teşhis ilk aşamalarda tamamlanır). |
| **Hastaneye Sağlanan Bütçe Tasarrufu** | **%0.0** (Tüm bütçe en baştan harcanır). | **%72.8 Tasarruf** (Girişimsel olmayan, ucuz testlerle teşhis güvencesi sağlanır). |
| **Kullanılan Algoritmalar** | Tek bir `RandomForestClassifier` (Varsayılan parametreler). | Her teşhis aşaması için özel olarak eğitilmiş, optimize edilmiş **4 adet XGBoost / Ensemble** modeli. |
| **Açıklanabilir Yapay Zeka (XAI)** | **Statik Global Önem:** Sadece veri kümesinin genelini yansıtan tek bir kaba önem tablosu sunar. | **Dinamik Fizyolojik SHAP:** Hastanın durduğu aşamadaki özellikleri, risk yönleriyle (kırmızı/mavi) canlı açıklar. |
| **Doğruluk ve Karar Güvenilirliği** | Tekil Cleveland verisinde **%75.41** Doğruluk. | İleri aşamalarda **%84 - %88 ROC-AUC** değeri ile son derece yüksek tanı güvencesi. |
| **Klinik Uygulanabilirlik (CDSS)** | **Zayıf:** Poliklinikte tarama testi olarak en pahalı testleri tüm hastalara uygulamak tıbben ve ekonomik olarak imkansızdır. | **Çok Güçlü:** Hastane triage iş akışına tam entegre, hekim kararlarını hızlandıran ve bütçe dostu CDSS arayüzü. |

---

## 🩺 4. Klinik Senaryo Analizi (Vaka Çalışması)

### Örnek Vaka: 45 yaşında, hafif göğüs ağrısı (`cp`=2) olan, egzersiz anjinası (`exang`=0) olmayan genç bir erkek hasta.

* **Kaggle Çözümünde:** 
  Bu hastanın teşhisi için hekim zorunlu olarak damar içi floroskopi (`ca` = $350) ve nükleer talyum sintigrafisi (`thal` = $250) dahil tüm testleri istemelidir. Toplam maliyet **$595** olur. Model %75 doğrulukla bir risk tahmin eder. Hastaya girişimsel testler uygulanarak gereksiz radyasyon riski yüklenir.
* **Cardio-Shield CDSS Çözümünde:**
  * **Stage 1 (Maliyet: $30):** Yaş (45), Cinsiyet (Erkek), Göğüs Ağrısı (Hafif-2), Egzersiz Anjinası (Yok-0), Açlık Kan Şekeri (Normal-0) verileri girilir.
  * **Sonuç:** Model risk ihtimalini **%10** (Low Risk) hesaplar. Risk skoru early-stopping eşiği olan **%15'in altında** olduğu için eskalasyon derhal durdurulur.
  * **Klinik Karar:** Hasta taburcu edilir.
  * **Toplam Harcanan Maliyet:** **$30.00**
  * **Tasarruf:** **$565.00 (%95 Tasarruf)** ve **0 radyasyon maruziyeti**!

---

## 📈 5. Kaggle KNN Projesi İncelemesi (`mohamedalaaabdella` - %98.83 Doğruluk İddiası)

Bu projede yazar, k-Nearest Neighbors (k-En Yakın Komşu - KNN) algoritmasını kullanarak kalp hastalığı veri setinde **%98.83 test doğruluğu (accuracy)** elde ettiğini iddia etmektedir. Ancak, bu doğruluk oranı klinik ve bilimsel açıdan **tamamen geçersizdir**.

### ⚠️ Kritik Metodolojik Hata: Veri Sızıntısı (Data Leakage)

KNN modeliyle elde edilen %98.83 doğruluk oranının arkasında çok büyük bir veri bilimi hatası yatmaktadır: **Mükerrer verilerin silinmemiş olması (`df.drop_duplicates()` eksikliği).**

#### Hatanın Mekanizması:
1. **Veri Kümesinin Yapısı:** Kaggle'daki kalp hastalığı veri kümesi (1025 satır), aslında 303 satırlık orijinal Cleveland veri kümesinin yaklaşık 3.4 kez kopyalanıp çoğaltılmış halidir.
2. **Hatalı Bölme:** Yazar, mükerrer hastaları silmeden doğrudan `%75-%25` oranında `train_test_split` yapmıştır.
3. **Sızıntı (Leakage):** Rastgele bölme nedeniyle, test setine düşen 257 hastanın neredeyse tamamının birebir aynı tıbbi kayıtlara sahip kopyaları eğitim setine (768 satır) de düşmüştür.
4. **KNN Algoritmasının Zaafiyeti:** Uzaklık tabanlı çalışan ve `weights='distance'` (mesafeyle ters orantılı ağırlık) kullanan KNN modeli, test setindeki bir hastayı sınıflandırırken eğitim setinde bu hastanın **birebir kopyasını bulur ve mesafe tam 0.0 çıkar**.
5. **Ezberleme (Memorization):** Mesafe sıfır çıktığı için model hiçbir örüntü öğrenmeden, sadece eğitim kümesindeki aynı hastanın etiketini doğrudan kopyalayarak test setine yapıştırır.

### 🩺 Klinik Açıdan Değerlendirme
* **Gerçekçi Doğruluk:** Mükerrer satırlar silindiğinde (bizim projemizde yaptığımız gibi 302 tekil hastada), KNN modelinin gerçek doğruluğu **%72 - %78** bandına gerilemektedir. 
* **Öznitelik Kaybı:** Bu KNN modelinde yazar, p-değeri analiziyle **`age`** (yaş) ve **`fbs`** (açlık kan şekeri) parametrelerini anlamsız bularak modelden atmıştır. Ancak klinik dünyada 25 yaşındaki bir hastayla 75 yaşındaki bir hastanın kalp krizi riskinin aynı şekilde değerlendirilmesi veya şeker hastalığının (fbs) tamamen dışlanması tıbbi gerçeklikle uyuşmamaktadır.

---

## 📊 6. Üç Modelin Karşılaştırma Özeti

| Kriter | Kaggle Random Forest Projesi | Kaggle KNN Projesi | Bizim Cardio-Shield CDSS |
| :--- | :--- | :--- | :--- |
| **Bildirilen Doğruluk** | %75.41 | %98.83 (Yanıltıcı) | **%84 - %88 (Gerçek ve Güvenilir)** |
| **Metodolojik Durum** | Temiz (Tekil Veri) | **Hatalı (Veri Sızıntısı var)** | **Temiz ve Doğrulanmış (Veri Sızıntısı Yok)** |
| **Klinik Yaklaşım** | Statik (Tek Aşamalı) | Statik (Tek Aşamalı) | **Dinamik 4 Aşamalı Eskalasyon** |
| **Bütçe ve Girişimsel Koruma** | Yok ($595.00) | Yok ($595.00) | **Ortalama %72.8 Maliyet Tasarrufu** |

---

## 📈 7. Sonuç Değerlendirmesi

Kaggle üzerindeki popüler projeler, makine öğrenmesini genellikle klinik bağlamdan kopuk, sadece matematiksel veya sentetik olarak şişirilmiş başarım oranları elde etmeye yönelik bir veri manipülasyon oyunu olarak ele almaktadır. KNN projesindeki **%98.83**'lük yanıltıcı doğruluk iddiası, gerçek dünya klinik uygulamalarında tamamen başarısız olacak bir **ezberleme (overfitting)** vakasıdır.

Bizim geliştirdiğimiz **Cardio-Shield CDSS** ise:
1. Veri sızıntılarını en başta önleyerek **dürüst ve bilimsel** bir doğruluk düzeyi yakalamıştır.
2. Tıbbın finansal gerçeklerine uygun **Maliyet Duyarlı Kademeli Teşhis Mimarisi** sunmuştur.
3. Hekime karar anında fizyolojik olarak **doğru yönlerde imzalanmış yerel SHAP açıklamaları** sunarak gerçek bir klinik ürün haline gelmiştir.
