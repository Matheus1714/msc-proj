import yaml
from pathlib import Path

DOCS_DIR = Path("docs")
MKDOCS_YML = Path("mkdocs.yaml")

def extract_title_from_md(md_path: Path) -> str | None:
    """Extrai o título do front matter YAML se existir."""
    try:
        with md_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines[0].strip() == "---":
            end = lines[1:].index("---\n") + 1
            front_matter = "".join(lines[1:end])
            meta = yaml.safe_load(front_matter)
            if isinstance(meta, dict) and "title" in meta:
                return str(meta["title"])
    except Exception:
        pass
    return None

def fallback_title(md_path: Path) -> str:
    """Cria um título legível a partir do nome do arquivo"""
    return md_path.stem.replace("-", " ").replace("_", " ").capitalize()

def build_nav_from_docs(base_dir: Path):
    nav = []

    index_md = base_dir / "index.md"
    if index_md.exists():
        title = extract_title_from_md(index_md) or "Início"
        nav.append({title: "index.md"})

    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            children = sorted(item.glob("*.md"))
            if children:
                section = []
                for child in children:
                    title = extract_title_from_md(child) or fallback_title(child)
                    section.append({title: str(child.relative_to(base_dir))})
                nav.append({item.name.capitalize(): section})

    return nav

def update_mkdocs_yaml(nav_data):
    base_config = {
        "site_name": "Docs Projeto de Mestrado",
        "site_url": "https://matheus1714.github.io/msc-proj/",
        "theme": {
            "name": "material",
            "palette": [
                {
                    "scheme": "slate",
                    "primary": "indigo",
                    "accent": "amber",
                    "toggle": {
                        "icon": "material/weather-sunny",
                        "name": "Mudar para tema claro",
                    },
                },
                {
                    "scheme": "default",
                    "primary": "indigo",
                    "accent": "indigo",
                    "toggle": {
                        "icon": "material/weather-night",
                        "name": "Mudar para tema escuro",
                    },
                }
            ],
        },
        "nav": nav_data
    }

    with open(MKDOCS_YML, "w", encoding="utf-8") as f:
        yaml.dump(
            base_config,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,  # <- essencial
            width=1000  # evita quebra de linha desnecessária
        )

    print("Arquivo mkdocs.yml gerado com sucesso.")

if __name__ == "__main__":
    nav = build_nav_from_docs(DOCS_DIR)
    update_mkdocs_yaml(nav)
