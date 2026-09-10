#!/usr/bin/env python3
"""
================================================================================
KEEP_ALIVE — envoltorio para desplegar ltc_smc_analyst.py en Render (free)
================================================================================
Lanza el script de análisis como proceso hijo y levanta un servidor HTTP
mínimo para que UptimeRobot pueda hacer ping y Render no duerma el servicio.

IMPORTANTE (corrección aplicada): se lanza el proceso hijo con la opción
-u (salida sin buffer). Sin esto, los mensajes de print() del bot pueden
quedar "atascados" en un buffer interno y no aparecer en los Logs de
Render hasta que se acumula suficiente texto o el proceso termina — dando
la falsa impresión de que el bot está colgado cuando en realidad sigue
trabajando por dentro.

Uso en el Procfile de Render:
    web: python keep_alive.py

Variables de entorno relevantes:
    PORT          La asigna Render automáticamente, no hay que tocarla.
    ANALYST_ARGS  Opcional, para cambiar los argumentos sin editar este
                  archivo, ej. "--interval 60 --watch 60 --telegram"
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
    # "-u" = salida sin buffer, para que los print() del bot aparezcan en
    # los Logs de Render al instante, en vez de quedar retenidos.
    return [sys.executable, "-u", "ltc_smc_analyst.py", *args]


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
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

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
