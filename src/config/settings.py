import os
from typing import Dict


def get_settings() -> Dict[str, str]:
    """
    Retorna as configurações básicas do projeto a partir de variáveis
    de ambiente, com valores padrão seguros.
    """
    return {
        "APP_NAME": os.getenv("APP_NAME", "Identificação de Vacas - Visão Computacional"),
        "ENV": os.getenv("ENV", "development"),
    }

