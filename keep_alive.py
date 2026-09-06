#!/usr/bin/env python3
"""
================================================================================
KEEP_ALIVE — envoltorio para desplegar ltc_smc_analyst.py en Render (free)
================================================================================
Render (plan gratuito) solo mantiene un servicio "despierto" si es de tipo
Web Service y recibe tráfico HTTP; los "Background Workers" sin servidor web
no están disponibles en el plan free. Nuestro bot original es un script en
bucle que no escucha peticiones HTTP, así que este archivo:

  1. Lanza ltc_smc_analyst.py como proceso hijo, con EXACTAMENTE los mismos
     argumentos que ya usabas en el worker de Railway (no se modifica ese
     archivo en absoluto).
  2. Si el proceso hijo llegara a caerse por cualquier motivo, lo reinicia
     solo (con una pequeña pausa) en vez de dejar el bot detenido.
  3. Levanta un servidor HTTP mínimo (solo librería estándar, sin
     dependencias nuevas) en el puerto que Render asigna vía la variable de
     entorno PORT. UptimeRobot debe apuntar a la URL pública de este
     servicio (la que te da Render) cada 10 minutos — Render duerme los
     servicios gratuitos tras 15 min sin tráfico, así que 10 min deja
     margen de sobra.

No cambia nada de la lógica SMC/ICT, el envío a Telegram, ni las variables
de entorno (SYMBOLS, TELEGRAM_BOT_TOKEN, etc.) — todas siguen funcionando
igual, configuradas en Render en vez de en Railway.

Uso en el Procfile de Render:
    web: python keep_alive.py

Variables de entorno relevantes:
    PORT               La asigna Render automáticamente, no hay que tocarla.
    ANALYST_ARGS        Opcional: para cambiar los argumentos del análisis
                        sin editar este archivo, ej.
                        "--interval 60 --watch 30 --telegram"
================================================================================
"""
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_ARGS = ["--interval", "60", "--watch", "60", "--telegram"]

_last_restart_count = 0
_lock = threading.Lock()


def build_analyst_cmd():
    custom = os.environ.get("ANALYST_ARGS", "").strip()
    args = custom.split() if custom else DEFAULT_ARGS
    return [sys.executable, "ltc_smc_analyst.py", *args]


def run_analyst_forever():
    global _last_restart_count
    cmd = build_analyst_cmd()
    print(f"[keep_alive] Lanzando: {' '.join(cmd)}", flush=True)
    while True:
        proc = subprocess.Popen(cmd)
        proc.wait()
        with _lock:
            _last_restart_count += 1
        print(
            f"[keep_alive] ltc_smc_analyst.py terminó con código "
            f"{proc.returncode}. Reiniciando en 10s (reinicio #{_last_restart_count})...",
            flush=True,
        )
        time.sleep(10)


class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _lock:
            restarts = _last_restart_count
        body = (
            f"ltc-smc-bot activo\n"
            f"reinicios del analista: {restarts}\n"
            f"hora UTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Silencia el log de cada ping de UptimeRobot para no llenar la
        # consola de Render con una línea cada 10 minutos.
        pass


def main():
    analyst_thread = threading.Thread(target=run_analyst_forever, daemon=True)
    analyst_thread.start()

    port = int(os.environ.get("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), PingHandler)
    print(f"[keep_alive] Servidor de ping escuchando en 0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
