"""
Lanzador de Escritorio Local para Audio ML Studio.

Inicia el servidor local de `app.py` en segundo plano en un puerto libre
y abre la interfaz dentro de una ventana nativa de escritorio (usando
`pywebview` si está disponible, o el motor nativo `--app` de Edge/Chrome
en Windows sin barras de navegador), cerrando el servidor automáticamente
al salir.

Uso:
    python desktop_app.py
"""

from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser


APP_TITLE = "Audio ML Studio — Procesamiento Acústico y Entrenamiento"
PROJECT_ROOT = Path(__file__).resolve().parent


def find_free_port(start_port: int = 8501) -> int:
    """Encuentra un puerto TCP libre en localhost a partir de `start_port`."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout_s: float = 30.0) -> bool:
    """Espera hasta que el servidor local responda HTTP 200."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def launch_native_app_window(url: str, server_proc: subprocess.Popen):
    """Abre la interfaz como ventana nativa de escritorio."""
    # Opción 1: Ventana nativa embebida con pywebview (si está instalado)
    try:
        import webview  # type: ignore

        webview.create_window(
            APP_TITLE,
            url,
            width=1360,
            height=860,
            min_size=(1024, 700),
        )
        webview.start()
        return
    except Exception:
        pass

    # Opción 2: Ventana nativa de escritorio sin marco de navegador (Edge / Chrome --app)
    browser_candidates = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = next((b for b in browser_candidates if b and Path(b).exists()), None)

    if browser_exe:
        print(f"Abriendo ventana de escritorio con: {Path(browser_exe).name}")
        subprocess.Popen([
            browser_exe,
            f"--app={url}",
            "--window-size=1360,860",
            "--disable-extensions",
        ])
    else:
        webbrowser.open(url)

    print("La aplicación está en ejecución. Presiona Ctrl+C en esta ventana para cerrar el servidor.")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        pass


def main():
    port = find_free_port(8501)
    url = f"http://127.0.0.1:{port}"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    server_proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )

    try:
        if not wait_for_server(url, timeout_s=35.0):
            print("Error: El servidor local tardó demasiado en iniciar.")
            return
        launch_native_app_window(url, server_proc)
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
