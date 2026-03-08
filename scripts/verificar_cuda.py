#!/usr/bin/env python
"""
Verifica se o PyTorch está usando CUDA e qual versão.
Se CUDA não estiver disponível, mostra como corrigir (reinstalar PyTorch com suporte GPU).

Uso: python scripts/verificar_cuda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    print("=== Verificação CUDA / PyTorch ===\n")

    # 1. Driver NVIDIA
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            print("  [OK] NVIDIA driver encontrado:")
            for line in out.stdout.strip().split("\n"):
                print(f"       {line}")
        else:
            print("  [AVISO] nvidia-smi não retornou dados. Atualize o driver NVIDIA.")
    except FileNotFoundError:
        print("  [AVISO] nvidia-smi não encontrado. Instale o driver NVIDIA (GeForce Game Ready / Studio).")
    except Exception as e:
        print(f"  [AVISO] Erro ao rodar nvidia-smi: {e}")

    # 2. PyTorch
    try:
        import torch
    except ImportError:
        print("\n  [FALHA] PyTorch não instalado. Instale: pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    print(f"\n  PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"  CUDA disponível: Sim")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA (PyTorch): {torch.version.cuda}")
    else:
        print("  CUDA disponível: Não")
        print("\n  Possível causa: PyTorch foi instalado na versão CPU (sem CUDA).")
        print("  Solução: reinstalar PyTorch com suporte CUDA (veja abaixo).\n")

    # 3. Instruções se não tiver CUDA
    if not cuda_available:
        print("  --- Como corrigir (Windows) ---")
        print("  1. Desative o ambiente virtual e reinstale o PyTorch com CUDA:")
        print("     pip uninstall torch torchvision torchaudio -y")
        print("     pip cache purge")
        print("     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121")
        print("  2. Se você instalou CUDA 12.4 ou 12.6, use cu124 em vez de cu121:")
        print("     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124")
        print("  3. Verifique a versão do CUDA no sistema: nvcc --version (se instalou o Toolkit)")
        print("  4. Reinicie o terminal/IDE após instalar e rode este script de novo.")
        print()
        sys.exit(1)

    print("\n  Concluído: GPU pronta para treino.\n")


if __name__ == "__main__":
    main()
