# 🚀 IF Tech - Git & Development Standards

## 🧠 Overview

IF Tech olarak tüm projelerimizde amaç sadece kod yazmak değil, ölçeklenebilir, sürdürülebilir ve ekip tabanlı yazılım geliştirme kültürü oluşturmaktır.

Bu nedenle tüm repository'lerde aşağıdaki Git workflow ve geliştirme standartları zorunludur.

## 🌿 Branch Strategy

### 📌 Main Branch Structure

| Branch | Açıklama |
| --- | --- |
| `main` | Production-ready kod (asla direkt push yapılmaz) |
| `develop` | Aktif geliştirme ana hattı |
| `feature/*` | Yeni özellik geliştirme |
| `bugfix/*` | Hata düzeltmeleri |
| `refactor/*` | Kod iyileştirme (logic değişmez) |
| `docs/*` | Dokümantasyon değişiklikleri |

> ⚠️ **Kritik Kural**
> * `main` branch → protected
> * Direkt push → ❌ yasak
> * Tüm değişiklikler → PR üzerinden

## 🔄 Development Workflow

### 🛠️ 1. Feature Başlatma
```bash
git checkout develop
git pull origin develop
git checkout -b feature/feature-name
```

### 💻 2. Geliştirme Süreci
Değişiklikleri yap ve commit at:
```bash
git status
git add .
git commit -m "feat: add login authentication system"
```

### 🔄 3. Develop Sync (Zorunlu)
PR açmadan önce:
```bash
git checkout develop
git pull origin develop
git checkout feature/feature-name
git merge develop
```
> ⚠️ Conflict varsa lokalde çözülür

### 🚀 4. Push İşlemi
```bash
git push origin feature/feature-name
```

### 📥 5. Pull Request (PR)
PR açarken:
- Açıklama net olmalı
- Yapılan değişiklikler yazılmalı
- İlgili issue bağlanmalı

**📌 Örnek:**
`Closes #12`

### 👀 6. Code Review
- En az 1 ekip üyesi onayı zorunlu
- Test edilmemiş kod merge edilmez
- Gerekirse değişiklik istenir

## 🧾 Commit Convention (Zorunlu)

Tüm commit mesajları aşağıdaki formatta olmalıdır:

| Type | Açıklama |
| --- | --- |
| `feat:` | Yeni özellik |
| `fix:` | Hata düzeltme |
| `docs:` | Dokümantasyon |
| `style:` | Format / UI değişiklik |
| `refactor:` | Kod iyileştirme |
| `test:` | Test ekleme |

### ✅ Örnek Commitler
- `feat: add cardio risk prediction model`
- `fix: resolve login token issue`
- `docs: update API documentation`
- `refactor: simplify authentication flow`

## 🧪 Quality Rules

### Kod Kalitesi
- Test edilmemiş kod → ❌ PR'a girmez
- Console log temizliği zorunlu
- Hardcoded değerlerden kaçınılır

### API Standards
Tüm API response'ları standart format kullanır:
```json
{
  "success": true,
  "data": {},
  "message": "Operation successful"
}
```

## 🛡️ Merge Rules
- `develop` → `main` sadece release durumunda
- Her merge sonrası versiyon artırılır
- Hotfix gerekiyorsa `hotfix/*` branch açılır

## 📊 Issue Management
Her görev bir issue olmalıdır:

| Label | Açıklama |
| --- | --- |
| `bug` | Hata |
| `feature` | Yeni özellik |
| `research` | Araştırma |
| `ai` | ML / AI işleri |
| `backend` | Backend task |
| `frontend` | UI task |

## ⚙️ CI/CD (Future Standard)
Tüm projelerde hedef:
- GitHub Actions
- Automatic build
- Lint check
- Test pipeline

## 📦 IF Tech Engineering Philosophy
IF Tech olarak geliştirme yaklaşımımız:
- "Working code" değil → "scalable system"
- "Single developer logic" değil → "team-based architecture"
- "fast hack" değil → "maintainable product"

### 🚀 Final Rule
Eğer bir kod PR'a giriyorsa, o kod production'a gitmeye hazırdır.