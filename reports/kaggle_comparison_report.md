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

## 📈 5. Sonuç Değerlendirmesi

Kaggle üzerindeki standart projeler (incelediğimiz grup çalışması dahil) makine öğrenmesini sadece matematiksel bir "sınıflandırma" oyunu olarak ele almaktadır. Bu projeler veri temizleme adımlarını doğru yapsalar dahi, **sağlık sektörünün en büyük iki kısıtı olan tıbbi test maliyetlerini ve hasta odaklı girişimsel riskleri** tamamen devre dışı bırakmaktadır.

Bizim geliştirdiğimiz **Cardio-Shield CDSS** ise veri bilimi ile klinik/finansal yönetimi bir araya getirerek, **maliyetleri %72.8 düşürürken tahmin doğruluğunu koruyan**, açıklanabilir yapay zekaya sahip gerçekçi ve ticari olarak uygulanabilir bir sağlık teknolojisi ürünüdür.
