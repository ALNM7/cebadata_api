"""Hace ping periodico al backend para que Render (plan free) no lo duerma.

Render free apaga el servicio tras ~15 min sin trafico, asi que se pega
cada 10 min al endpoint de salud.

Uso:
    python keep_alive.py
    python keep_alive.py --url https://otro.onrender.com/api/salud --intervalo 600
"""
import argparse
import time
import urllib.error
import urllib.request
from datetime import datetime

URL_POR_DEFECTO = "https://cebadata-api.onrender.com/api/salud"
INTERVALO_POR_DEFECTO = 600  # segundos (10 min)


def ping(url: str) -> None:
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inicio = time.time()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            ms = int((time.time() - inicio) * 1000)
            print(f"[{ahora}] OK {resp.status} en {ms} ms")
    except urllib.error.HTTPError as e:
        print(f"[{ahora}] HTTP {e.code}")
    except Exception as e:
        print(f"[{ahora}] Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ping para mantener despierto el backend")
    parser.add_argument("--url", default=URL_POR_DEFECTO)
    parser.add_argument("--intervalo", type=int, default=INTERVALO_POR_DEFECTO)
    args = parser.parse_args()

    print(f"Ping a {args.url} cada {args.intervalo} s. Ctrl+C para parar.")
    while True:
        ping(args.url)
        time.sleep(args.intervalo)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDetenido.")
