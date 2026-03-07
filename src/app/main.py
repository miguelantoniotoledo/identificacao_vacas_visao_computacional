from src.config import AppParams, get_params, get_settings


def main() -> None:
    settings = get_settings()
    project_name = settings.get("APP_NAME", "Python Base Project")
    env = settings.get("ENV", "development")
    params: AppParams = get_params()

    print(f"Rodando {project_name} no ambiente '{env}'.")
    print(
        "Parâmetros:",
        f"modelo={params.model_name},",
        f"image_size={params.image_size},",
        f"confidence_threshold={params.confidence_threshold}",
    )


if __name__ == "__main__":
    main()

