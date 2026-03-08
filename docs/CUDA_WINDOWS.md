# CUDA no Windows: ajustar quando PyTorch não enxerga a GPU

Se você instalou o driver NVIDIA, CUDA Toolkit e Microsoft Visual C++ mas o treino ainda roda em CPU (`torch.cuda.is_available()` retorna `False`), o motivo mais comum é **PyTorch instalado na versão CPU**.

## 1. Verificar o que está acontecendo

No projeto, com o ambiente ativado:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/verificar_cuda.py
```

O script mostra se o driver NVIDIA está OK e se o PyTorch está com CUDA disponível.

## 2. Reinstalar PyTorch com suporte CUDA

O PyTorch precisa ser instalado explicitamente com o índice das builds **cuXXX** (GPU). Só o `requirements.txt` com `--extra-index-url` às vezes não basta se o `torch` já foi instalado antes como CPU.

### Passo a passo

1. **Desinstalar** a versão atual do PyTorch:

   ```powershell
   pip uninstall torch torchvision torchaudio -y
   ```

2. **Limpar o cache** do pip (evita reinstalar o mesmo pacote CPU):

   ```powershell
   pip cache purge
   ```

3. **Instalar** PyTorch com CUDA 12.1 (recomendado na maioria dos PCs):

   ```powershell
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
   ```

4. Se você instalou **CUDA 12.4** ou **12.6** no sistema e quiser alinhar:

   ```powershell
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
   ```

5. **Fechar e reabrir** o terminal (ou o Cursor) e rodar de novo:

   ```powershell
   python scripts/verificar_cuda.py
   ```

## 3. Conferir driver e CUDA no sistema

- **Driver NVIDIA:** abra um CMD/PowerShell e rode `nvidia-smi`. Deve listar a GPU e a versão do driver. Se não aparecer nada, instale ou atualize o [driver NVIDIA](https://www.nvidia.com/Download/index.aspx).
- **CUDA Toolkit (opcional):** o PyTorch já traz as libs CUDA necessárias. O Toolkit é útil para desenvolvimento. Para ver a versão: `nvcc --version` (só funciona se o Toolkit estiver no PATH).

## 4. Usar o projeto em CPU

Se não tiver GPU ou quiser forçar CPU no treino, no `config.yaml`:

```yaml
training:
  device: "cpu"
```

O treino será mais lento, mas funciona.
