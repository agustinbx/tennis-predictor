"""
Helper para inicializar Chrome con la versión correcta.

undetected_chromedriver a veces descarga un ChromeDriver que no
coincide con la versión instalada de Chrome. Este módulo detecta
la versión real y la pasa explícitamente.
"""
import subprocess
import re
import sys
import undetected_chromedriver as uc


def get_chrome_version() -> int:
    """
    Detecta la versión mayor de Chrome instalada en el sistema.

    Returns:
        int: versión mayor (ej: 147 para Chrome 147.0.7727.138)
    """
    # Windows: buscar en el registro o en la ruta estándar
    cmd = r'reg query "HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon" /v version 2>nul'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        match = re.search(r'(\d+)\.\d+\.\d+\.\d+', result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass

    # Fallback: buscar en Program Files
    cmd = r'reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Google\Chrome\BLBeacon" /v version 2>nul'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        match = re.search(r'(\d+)\.\d+\.\d+\.\d+', result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass

    # Último fallback: dejar que undetected_chromedriver lo detecte
    return None


def create_chrome_driver(options=None, headless=False):
    """
    Crea una instancia de undetected_chromedriver.Chrome
    con la versión correcta de Chrome detectada automáticamente.

    Args:
        options: ChromeOptions existentes (opcional)
        headless: Si True, ejecuta sin interfaz gráfica

    Returns:
        undetected_chromedriver.Chrome instance
    """
    if options is None:
        options = uc.ChromeOptions()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--start-maximized")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")

    version = get_chrome_version()

    kwargs = {"options": options}
    if version is not None:
        print(f"[CONFIG] Chrome version {version} detectado")
        kwargs["version_main"] = version
    else:
        print("[WARN] No se pudo detectar version de Chrome, usando autodeteccion")

    try:
        driver = uc.Chrome(**kwargs)
        return driver
    except Exception as e:
        if version is not None:
            print(f"[WARN] Error con version {version}: {e}")
            print("[CONFIG] Reintentando sin version especifica...")
            kwargs.pop("version_main", None)
            return uc.Chrome(**kwargs)
        raise