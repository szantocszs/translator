# hu_dub — Magyar Szinkron Generáló

MP4 videókhoz és MP3/MP4 audio fájlokhoz automatizált transzkripciót, magyar szinkront és feliratot készítő CLI alkalmazás.

## Módok

| Mód | Bemenet | Leírás | Kimenet |
|-----|---------|--------|---------|
| `transcribe` | MP3/MP4 | Csak transzkripció Whisper-rel (magyar/angol/bármilyen nyelv) | SRT + tiszta TXT |
| `subtitle` | MP4 | Transzkripció + fordítás + SRT fájlok + beágyazott feliratok | MP4 + SRT-k |
| `dub` | MP4 | Teljes szinkron: felirat + magyar hangsáv (alapértelmezett) | MP4 + SRT-k |

### Transcribe mód

A `transcribe` mód kizárólag a hang átírására szolgál — **nem fordít, nem generál szinkront**. Ideális:
- Magyar nyelvű podcast-ok, előadások, meetingek átírására
- Bármilyen nyelvű audio/videó szövegének kinyerésére
- SRT felirat generálására időbélyegekkel
- Tiszta szöveg exportra (időbélyegek nélkül) szerkesztéshez, jegyzeteléshez

A Whisper `large-v3` modell az alapértelmezett transcribe módban, mert ez kezeli legjobban a nem angol nyelveket (pl. magyar). A nyelv automatikusan felismerhető, vagy `--language hu` kapcsolóval explicit megadható.

## TTS módszerek (dub módhoz)

| Módszer | Leírás | GPU kell? |
|---------|--------|-----------|
| `clone` | XTTS-v2 hangklónozás a beszélő eredeti hangjával | Igen (CUDA) |
| `edge` | Edge-TTS általános magyar hang (hu-HU-TamasNeural) | Nem |

## Fordítás backend-ek (subtitle/dub módhoz)

| Backend | Leírás | Költség |
|---------|--------|--------|
| `openai` | OpenAI GPT API (gpt-4o, gpt-4o-mini, stb.) | Fizetős |
| `ollama` | Helyi Ollama LLM (gemma3, qwen3, deepseek, stb.) | Ingyenes |

### Ajánlott Ollama modellek magyar fordításhoz (32GB VRAM)

| Modell | Méret | Minőség | Sebesség |
|--------|-------|---------|----------|
| `gemma3:27b` | ~16 GB | ★★★★★ | ★★★ |
| `qwen3:30b-a3b` | ~18 GB | ★★★★★ | ★★★ |
| `deepseek-v3:32b` | ~20 GB | ★★★★ | ★★ |
| `towerinstruct:13b` | ~8 GB | ★★★★ | ★★★★ |
| `gemma3:12b` | ~7 GB | ★★★ | ★★★★★ |
| `mistral:7b` | ~4 GB | ★★★ | ★★★★★ |

## Telepítés

```bash
pip install coqui-tts[codec] openai openai-whisper pydub tqdm edge-tts soundfile
```

## Használat

### Transzkripció (MP3/MP4 → SRT + szöveg)
```bash
# Automatikus nyelvfelismerés
python hu_dub/main.py -i "podcast.mp3" --mode transcribe

# Magyar nyelv explicit megadása (large-v3 az alapértelmezett transcribe módban)
python hu_dub/main.py -i "eloadas.mp4" --mode transcribe --language hu

# Angol nyelvű videó gyors átírása kisebb modellel
python hu_dub/main.py -i "talk.mp4" --mode transcribe --language en -w turbo

# Batch: egy könyvtár összes mp3/mp4 fájlja
python hu_dub/main.py -i ./recordings/ --batch --mode transcribe --language hu
```

### Csak feliratok (gyors, nincs szinkron hang)
```bash
python hu_dub/main.py -i "video.mp4" --mode subtitle
```

### Szinkron Edge-TTS-sel (általános magyar hang, gyors)
```bash
python hu_dub/main.py -i "video.mp4" --mode dub --tts-method edge
```

### Szinkron hangklónozással (XTTS-v2, a beszélő hangjával)
```bash
python hu_dub/main.py -i "video.mp4" --mode dub --tts-method clone
```

### Ollama-val fordítás (helyi, ingyenes)
```bash
python hu_dub/main.py -i "video.mp4" --translator ollama --ollama-model gemma3:27b
```

### Batch feldolgozás
```bash
python hu_dub/main.py -i ./videos/ --batch -w large-v3 --mode dub --tts-method edge
```

### Összes opció
```bash
python hu_dub/main.py \
  -i "video.mp4"                # Bemeneti fájl vagy könyvtár (MP4, transcribe módban MP3 is)
  -o "./output/"                # Kimeneti könyvtár
  --mode dub                    # transcribe | subtitle | dub
  --language hu                 # Audio nyelve (transcribe módhoz; subtitle/dub = en fix)
  --tts-method clone            # clone | edge (dub módhoz)
  -w large-v3                   # Whisper modell (transcribe default: large-v3, egyébként: medium)
  --translator openai           # openai | ollama (subtitle/dub módhoz)
  --openai-api-key "sk-..."     # OpenAI API kulcs
  --openai-model gpt-4o         # OpenAI modell
  --ollama-model gemma3:27b     # Ollama modell
  --ollama-url http://...       # Ollama szerver URL
  --voice-sample-sec 30         # Hangminta (clone módhoz)
  --batch                       # Könyvtár mód
  --skip-existing               # Már kész fájlok kihagyása
  --keep-temp                   # Debug: temp fájlok megtartása
  --verbose                     # Részletes log
```

## OpenAI API kulcs

Keresési sorrend (csak subtitle/dub módban kell, ha OpenAI a fordító):
1. `--openai-api-key` parancssori argumentum
2. `AI__OpenAiKey` környezeti változó
3. `OPENAI_API_KEY` környezeti változó

## Kimeneti fájlok

| Fájl | Mód | Tartalom |
|------|-----|----------|
| `XYZ.srt` | transcribe | Felirat időbélyegekkel |
| `XYZ.txt` | transcribe | Tiszta szöveg (időbélyegek nélkül) |
| `XYZ_HU.mp4` | subtitle/dub | Videó feliratokkal és/vagy magyar hangsávval |
| `XYZ_EN.srt` | subtitle/dub | Angol felirat fájl |
| `XYZ_HU.srt` | subtitle/dub | Magyar felirat fájl |

## Whisper modellek

| Modell | Méret | Sebesség | Pontosság | Megjegyzés |
|--------|-------|----------|-----------|------------|
| tiny | 39 MB | ★★★★★ | ★★ | Csak angol |
| base | 74 MB | ★★★★ | ★★★ | |
| small | 244 MB | ★★★ | ★★★★ | |
| **medium** | 769 MB | ★★ | ★★★★ | Default (subtitle/dub) |
| **large-v3** | 1.55 GB | ★ | ★★★★★ | Default (transcribe) — legjobb magyar támogatás |
| turbo | 809 MB | ★★★ | ★★★★★ | Gyors + pontos, de angol-centrikus |

> **Megjegyzés:** Magyar nyelvű transzkripciőhoz a `large-v3` modell ajánlott, mert ez kezeli legjobban a nem angol nyelveket. Az `.en` végű modellek (tiny.en, base.en, stb.) kizárólag angol nyelvűek.
