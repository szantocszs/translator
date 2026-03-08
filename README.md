# hu_dub — Magyar Szinkron Generáló

Angol nyelvű MP4 videókhoz automatizált magyar szinkront és feliratot készítő CLI alkalmazás.

## Módok

| Mód | Leírás |
|-----|--------|
| `subtitle` | Csak felirat: transzkripció + fordítás + SRT fájlok + beágyazott feliratok |
| `dub` | Teljes szinkron: felirat + magyar hangsáv (alapértelmezett) |

## TTS módszerek (dub módhoz)

| Módszer | Leírás | GPU kell? |
|---------|--------|-----------|
| `clone` | XTTS-v2 hangklónozás a beszélő eredeti hangjával | Igen (CUDA) |
| `edge` | Edge-TTS általános magyar hang (hu-HU-TamasNeural) | Nem |

## Fordítás backend-ek

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
  -i "video.mp4"                # Bemeneti fájl vagy könyvtár
  -o "./output/"                # Kimeneti könyvtár
  --mode dub                    # subtitle | dub
  --tts-method clone            # clone | edge (dub módhoz)
  -w medium                     # Whisper modell
  --translator openai           # openai | ollama
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

Keresési sorrend:
1. `--openai-api-key` parancssori argumentum
2. `AI__OpenAiKey` környezeti változó
3. `OPENAI_API_KEY` környezeti változó

## Kimeneti fájlok

| Fájl | Tartalom |
|------|----------|
| `XYZ_HU.mp4` | Videó feliratokkal (subtitle mód) vagy feliratokkal + magyar hangsávval (dub mód) |
| `XYZ_EN.srt` | Angol felirat fájl |
| `XYZ_HU.srt` | Magyar felirat fájl |

## Whisper modellek

| Modell | Méret | Sebesség | Pontosság |
|--------|-------|----------|-----------|
| tiny | 39 MB | ★★★★★ | ★★ |
| base | 74 MB | ★★★★ | ★★★ |
| small | 244 MB | ★★★ | ★★★★ |
| **medium** | 769 MB | ★★ | ★★★★ |
| large-v3 | 1.55 GB | ★ | ★★★★★ |
| turbo | 809 MB | ★★★ | ★★★★★ |
