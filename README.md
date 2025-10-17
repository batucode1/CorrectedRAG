# CorrectedRAG

**CorrectedRAG**, LangChain ve LangGraph kullanılarak oluşturulmuş, modüler yapıya sahip bir Retrieval-Augmented Generation (RAG) projesidir.  
Proje, metin verilerini vektör haline getirip Chroma veritabanında saklar, ardından kullanıcının sorduğu sorulara bu veriler üzerinden yanıt üretir.  
GraphState mimarisi sayesinde her işlem düğüm (node) bazında yönetilir ve akış kontrolü esnektir.

---

## Gereksinimler

- Python 3.10 veya üzeri  
- OpenAI API anahtarı  
- Gerekli bağımlılıkların kurulumu:
```bash
pip install -r requirements.txt
```

---


## Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/batucode1/CorrectedRAG.git
cd CorrectedRAG
```

### 2. Sanal Ortam Oluşturun
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# veya
.venv\Scripts\activate   # Windows
```

### 3. Gerekli Kütüphaneleri Kurun
```bash
pip install -r requirements.txt
```

### 4. Ortam Değişkenlerini Tanımlayın
Kök dizine `.env` dosyası ekleyin:

```env
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini
PERSIST_DIR=./data/chroma
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## Ingestion (Veri Hazırlama)

Belirli bir web sayfasını yükleyip Chroma veritabanına kaydetmek için:

```bash
python ingestion.py --urls "https://lilianweng.github.io/posts/2023-06-23-agent" --persist-dir "./data/chroma"
```

Bu işlem:
1. Sayfayı indirir.  
2. `RecursiveCharacterTextSplitter` ile metni parçalar.  
3. `OpenAIEmbeddings` ile vektörlere dönüştürür.  
4. Veriyi `Chroma` veritabanına yazar.

---

## Sorgu (Query) Çalıştırma

Hazırlanan veritabanı üzerinde soru sormak için:

```bash
python main.py --question "Yapay zeka ajanı nedir?"
```

Program:
1. Soru için uygun belgeleri Chroma’dan getirir.  
2. LangGraph üzerinden GraphState akışını başlatır.  
3. Retrieval → Grading → Generation zincirini yürütür.  
4. Nihai yanıtı üretir ve terminalde gösterir.

---

## GraphState Akışı

GraphState, her adımı bir düğüm (Node) olarak tanımlar.  
Bu düğümler arasında veri akışı aşağıdaki sırayla gerçekleşir:

1. **Retrieve Node:** Soruya uygun belgeleri getirir.  
2. **Grade Documents Node:** Belgeleri kaliteye göre değerlendirir.  
3. **Generate Node:** Uygun belgeleri kullanarak model yanıtını üretir.  
4. **Answer Grader Chain:** Üretilen cevabın tutarlılığını ve doğruluğunu kontrol eder.  
5. **Router:** Gerekirse yeniden sorgu veya ek retrieval adımı başlatır.  

Tüm bu akış `graph.py` ve `state.py` dosyaları üzerinden yönetilir.

