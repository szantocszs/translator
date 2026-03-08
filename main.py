#!/usr/bin/env python3
"""
hu_dub — Magyar szinkron generáló CLI alkalmazás

Módok:
  subtitle  — Csak felirat: transzkripció + fordítás + SRT fájlok + beágyazott feliratok
  dub       — Teljes szinkron: felirat + magyar hangsáv (hangklónozással vagy Edge-TTS-sel)

TTS módszerek (dub módhoz):
  clone     — XTTS-v2 hangklónozás az eredeti beszélő hangjával
  edge      — Edge-TTS általános magyar hang (gyorsabb, nincs GPU igény)

Fordítás backend-ek:
  openai    — OpenAI GPT API (fizetős, legjobb minőség)
  ollama    — Helyi Ollama LLM (ingyenes, GPU-n fut)
"""

import argparse
import logging
import os
import sys
import glob as glob_module

from transcriber import AVAILABLE_MODELS
from translator import RECOMMENDED_OLLAMA_MODELS
from pipeline import Pipeline


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger("hu_dub")
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_openai_key(args_key: str = None) -> str:
    """OpenAI API kulcs: arg > AI__OpenAiKey env > OPENAI_API_KEY env."""
    if args_key:
        return args_key
    key = os.environ.get("AI__OpenAiKey", "")
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    return ""


def find_mp4_files(path: str, batch: bool) -> list:
    if os.path.isfile(path):
        if not path.lower().endswith(".mp4"):
            print(f"Hiba: A fájl nem MP4: {path}", file=sys.stderr)
            sys.exit(1)
        return [path]
    if os.path.isdir(path):
        if not batch:
            print(f"Hiba: '{path}' könyvtár. Használd a --batch opciót.", file=sys.stderr)
            sys.exit(1)
        files = sorted(glob_module.glob(os.path.join(path, "*.mp4")))
        files = [f for f in files if not f.endswith("_HU.mp4")]
        if not files:
            print(f"Hiba: Nem találtam MP4 fájlokat: {path}", file=sys.stderr)
            sys.exit(1)
        return files
    print(f"Hiba: Nem létezik: {path}", file=sys.stderr)
    sys.exit(1)


def main():
    ollama_models_help = ", ".join(RECOMMENDED_OLLAMA_MODELS)

    parser = argparse.ArgumentParser(
        prog="hu_dub",
        description="Magyar szinkron generáló — angol MP4 videókhoz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Példák:
  # Csak feliratok (nincs szinkron hang)
  python hu_dub/main.py -i "video.mp4" --mode subtitle

  # Szinkron Edge-TTS-sel (gyors, általános magyar hang)
  python hu_dub/main.py -i "video.mp4" --mode dub --tts-method edge

  # Szinkron hangklónozással (XTTS-v2, a beszélő hangjával)
  python hu_dub/main.py -i "video.mp4" --mode dub --tts-method clone

  # Ollama-val fordítás (helyi LLM, ingyenes)
  python hu_dub/main.py -i "video.mp4" --translator ollama --ollama-model gemma3:27b

  # Batch feldolgozás
  python hu_dub/main.py -i ./videos/ --batch -w large-v3 --mode dub --tts-method edge

Ajánlott Ollama modellek magyar fordításhoz (32GB VRAM):
  {ollama_models_help}
        """,
    )

    # Alap paraméterek
    parser.add_argument("-i", "--input", required=True, help="Bemeneti MP4 fájl vagy könyvtár")
    parser.add_argument("-o", "--output", help="Kimeneti könyvtár (default: bemeneti mappa)")
    parser.add_argument("--batch", action="store_true", help="Könyvtár mód: minden MP4 feldolgozása")
    parser.add_argument("--skip-existing", action="store_true", help="Már létező _HU.mp4 kihagyása")
    parser.add_argument("--keep-temp", action="store_true", help="Temp fájlok megtartása (debug)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Részletes log")

    # Mód
    parser.add_argument(
        "--mode", choices=["subtitle", "dub"], default="dub",
        help="Feldolgozási mód: subtitle (csak felirat) | dub (felirat + szinkron) [default: dub]",
    )

    # Whisper
    parser.add_argument(
        "-w", "--whisper-model", default="medium", choices=AVAILABLE_MODELS,
        help="Whisper modell a transzkripciőhoz [default: medium]",
    )

    # Fordítás
    parser.add_argument(
        "--translator", choices=["openai", "ollama"], default="openai",
        help="Fordítás backend: openai (GPT API) | ollama (helyi LLM) [default: openai]",
    )
    parser.add_argument(
        "--openai-api-key", help="OpenAI API kulcs (default: AI__OpenAiKey / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-model", default="gpt-4o", help="OpenAI modell [default: gpt-4o]",
    )
    parser.add_argument(
        "--ollama-model", default="gemma3:27b",
        help=f"Ollama modell [default: gemma3:27b]. Ajánlott: {ollama_models_help}",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama szerver URL [default: http://localhost:11434]",
    )

    # TTS (dub módhoz)
    parser.add_argument(
        "--tts-method", choices=["clone", "edge"], default="clone",
        help="TTS módszer: clone (XTTS-v2 hangklón) | edge (Edge-TTS általános hang) [default: clone]",
    )
    parser.add_argument(
        "--voice-sample-sec", type=int, default=30,
        help="Hangminta hossza a klónozáshoz [default: 30s, csak clone módhoz]",
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    # Validáció
    api_key = ""
    if args.translator == "openai":
        api_key = get_openai_key(args.openai_api_key)
        if not api_key:
            print(
                "Hiba: Nincs OpenAI API kulcs.\n"
                "Megadási módok: --openai-api-key, AI__OpenAiKey env var, OPENAI_API_KEY env var\n"
                "Vagy használj Ollama-t: --translator ollama",
                file=sys.stderr,
            )
            sys.exit(1)

    mp4_files = find_mp4_files(args.input, args.batch)

    if args.output:
        output_dir = args.output
    elif os.path.isfile(args.input):
        output_dir = os.path.dirname(os.path.abspath(args.input))
    else:
        output_dir = os.path.abspath(args.input)

    # Info kiírás
    logger.info("hu_dub — Magyar szinkron generáló")
    logger.info(f"  Mód: {args.mode}")
    logger.info(f"  Fájlok: {len(mp4_files)} db")
    logger.info(f"  Whisper: {args.whisper_model}")
    trans_info = f"{args.translator} ({args.openai_model})" if args.translator == "openai" else f"{args.translator} ({args.ollama_model})"
    logger.info(f"  Fordítás: {trans_info}")
    if args.mode == "dub":
        logger.info(f"  TTS: {args.tts_method}" + (f" (hangminta: {args.voice_sample_sec}s)" if args.tts_method == "clone" else ""))
    logger.info(f"  Kimenet: {output_dir}")

    success = 0
    failed = 0
    skipped = 0

    for i, mp4_file in enumerate(mp4_files):
        base = os.path.splitext(os.path.basename(mp4_file))[0]
        out_file = os.path.join(output_dir, f"{base}_HU.mp4")

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(mp4_files)}] {os.path.basename(mp4_file)}")
        logger.info(f"{'='*60}")

        if args.skip_existing and os.path.exists(out_file):
            logger.info(f"  Kihagyva (már létezik): {out_file}")
            skipped += 1
            continue

        try:
            pipeline = Pipeline(
                input_mp4=mp4_file,
                output_dir=output_dir,
                mode=args.mode,
                whisper_model=args.whisper_model,
                translator=args.translator,
                openai_api_key=api_key,
                openai_model=args.openai_model,
                ollama_model=args.ollama_model,
                ollama_url=args.ollama_url,
                tts_method=args.tts_method,
                voice_sample_sec=args.voice_sample_sec,
                keep_temp=args.keep_temp,
            )
            pipeline.run()
            success += 1
        except Exception as e:
            logger.error(f"  ❌ Feldolgozás sikertelen: {e}")
            failed += 1
            if not args.batch:
                sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"Összesítés: {success} sikeres, {failed} sikertelen, {skipped} kihagyva")
    logger.info(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
