from pathlib import Path
import yaml

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
    return md_path.stem

def build_nav_from_docs(base_dir: Path):
    nav = []

    index_md = base_dir / "index.md"
    if index_md.exists():
        title = extract_title_from_md(index_md) or "Início"
        nav.append({title: "index.md"})

    sections = {}

    # Coleta todos os arquivos Markdown
    for md_path in sorted(base_dir.rglob("*.md")):
        rel_path = md_path.relative_to(base_dir)
        if rel_path == Path("index.md"):
            continue

        parts = rel_path.parts
        *folders, filename = parts

        current = sections
        for folder in folders:
            current = current.setdefault(folder, {})

        # Detecta se é index.md
        if filename == "index.md":
            current["__section_title__"] = extract_title_from_md(md_path) or fallback_title(md_path.parent)
            current["__index__"] = str(rel_path)
        else:
            title = extract_title_from_md(md_path) or fallback_title(md_path)
            current[title] = str(rel_path)

    def dict_to_nav(d):
        items = []
        for key, value in d.items():
            if key in ["__section_title__", "__index__"]:
                continue
            if isinstance(value, dict):
                section_title = value.get("__section_title__", key.capitalize())
                children = dict_to_nav(value)
                index_path = value.get("__index__")
                if index_path:
                    children.insert(0, {section_title: index_path})
                    items.append({section_title: children})
                else:
                    items.append({section_title: children})
            else:
                items.append({key: value})
        return items

    nav.extend(dict_to_nav(sections))
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
        "markdown_extensions": [
            "admonition",
            "pymdownx.details",
            "pymdownx.superfences",
        ],
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
