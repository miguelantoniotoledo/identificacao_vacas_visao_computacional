# Configuração para GPU NVIDIA (incluindo RTX 5060)

A **RTX 5060** usa a arquitetura **Blackwell** (compute capability sm_120). Versões antigas do PyTorch (por exemplo com CUDA 12.1) **não** incluem suporte a essa arquitetura e podem gerar:

- `torch.cuda.is_available(): False`, ou  
- `RuntimeError: CUDA error: no kernel image is available for execution on the device`

Siga os passos abaixo para usar a GPU corretamente.

---

## 1. Driver NVIDIA

- **Recomendado:** driver **576.x ou mais recente** (suporte a CUDA 12.8).
- Verifique no [site da NVIDIA](https://www.nvidia.com/Download/index.aspx) ou com:

```powershell
nvidia-smi
```

Anote a versão do driver. Se for muito antiga, atualize antes de instalar o PyTorch.

---

## 2. PyTorch com CUDA 12.8 (obrigatório para RTX 50xx)

**Não** use o build “CPU” nem CUDA 12.1/12.4 para RTX 5060. Use **CUDA 12.8** (índice `cu128`).

Em um ambiente virtual ativado (por exemplo `.venv`):

```powershell
# Desinstalar torch/torchvision/torchaudio antigos (se existirem)
pip uninstall torch torchvision torchaudio -y

# Instalar PyTorch 2.7.x com CUDA 12.8 (suporte a Blackwell / sm_120)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Depois instale o resto do projeto:

```powershell
pip install -r requirements.txt
```

(O `requirements.txt` usa `--extra-index-url` cu128, então as outras dependências não vão sobrescrever o torch com versão CPU.)

---

## 3. Conferir se a GPU está sendo usada

```powershell
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Arquiteturas:', torch.cuda.get_arch_list())"
```

Esperado em máquina com RTX 5060 e instalação correta:

- `CUDA disponível: True`
- `GPU: NVIDIA GeForce RTX 5060` (ou nome similar)
- Na lista de arquiteturas deve aparecer **sm_120** (ou compatível).

Se `sm_120` não aparecer, o PyTorch instalado ainda não é o build cu128; repita o passo 2.

---

## 4. Config do projeto

No `config.yaml`, o treino já pode usar GPU com:

```yaml
training:
  device: "0"   # GPU 0
```

Se você tiver só uma GPU, `"0"` é o correto. Com múltiplas GPUs pode usar, por exemplo, `"0,1"`.

---

## Resumo rápido (RTX 5060)

| Item            | Recomendação                          |
|-----------------|----------------------------------------|
| Driver NVIDIA   | 576.x ou mais novo                     |
| PyTorch         | 2.7.x com **cu128** (CUDA 12.8)        |
| Comando install | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| config.yaml     | `training.device: "0"`                 |

Referências úteis:

- [PyTorch – Get Started Locally](https://pytorch.org/get-started/locally/)
- [PyTorch – Previous Versions](https://pytorch.org/get-started/previous-versions/) (se precisar de uma versão específica)
