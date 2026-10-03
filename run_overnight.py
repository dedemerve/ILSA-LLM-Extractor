"""
articles_full için kalan 490 kaydı tamamlar.
Günlük limit dolunca gece UTC 00:05 sıfırlanmasını bekler.
"""
import subprocess, time, datetime, pathlib, json, sys

PROJECT = pathlib.Path(__file__).parent
SCRIPT  = PROJECT / 'scripts' / 'generate_primary_outcome.py'
CACHE   = PROJECT / 'outputs' / 'primary_outcome_cache.json'
LOG     = pathlib.Path('/tmp/gen_po.log')
TOTAL   = 1756

def cache_len():
    try:
        return len(json.loads(CACHE.read_text()))
    except Exception:
        return 0

def seconds_until_reset():
    now   = datetime.datetime.utcnow()
    reset = (now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    return max(60, (reset - now).total_seconds())

run = 1
while True:
    n = cache_len()
    if n >= TOTAL:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ ZATEN TAMAMLANDI {n}/{TOTAL}", flush=True)
        sys.exit(0)

    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Çalıştırma #{run} — cache: {n}/{TOTAL}", flush=True)

    with open(LOG, 'w') as logf:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT)],
            stdout=logf, stderr=logf,
            cwd=str(PROJECT)
        )

    prev_cache = cache_len()
    while proc.poll() is None:
        time.sleep(30)
        log_text = LOG.read_text() if LOG.exists() else ''

        daily_hit = ('tokens per day' in log_text or 'TPD' in log_text
                     or 'per day' in log_text.lower() or 'GenerateRequestsPerDay' in log_text)
        if daily_hit:
            wait = seconds_until_reset()
            reset_local = datetime.datetime.now() + datetime.timedelta(seconds=wait)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Günlük limit doldu — {reset_local.strftime('%H:%M')} bekleniyor ({wait/3600:.1f} saat)", flush=True)
            proc.terminate()
            proc.wait()
            time.sleep(wait)
            break

        current = cache_len()
        delta = current - prev_cache
        prev_cache = current
        pct = current / TOTAL * 100
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {current}/{TOTAL} ({pct:.0f}%)  +{delta} yeni", flush=True)

        if current >= TOTAL:
            proc.wait()
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ TAMAMLANDI!", flush=True)
            sys.exit(0)

    if cache_len() >= TOTAL:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ TAMAMLANDI!", flush=True)
        sys.exit(0)

    run += 1
