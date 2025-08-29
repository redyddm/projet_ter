from pathlib import Path
import unicodedata

# Dossiers/fichiers à ignorer
IGNORE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
IGNORE_FILES = {".DS_Store"}

# Seuil de nombre de fichiers avant de résumer le dossier
MAX_FILES_LIST = 5

def normalize_filename(name: str) -> str:
    """Normalise les accents pour un affichage correct"""
    return unicodedata.normalize("NFC", name)

def walk_dir(path: Path, prefix="") -> str:
    """Parcourt un dossier et construit une structure Markdown"""
    content = ""
    for item in sorted(path.iterdir()):
        name = normalize_filename(item.name)
        if item.is_dir():
            if name in IGNORE_DIRS:
                continue
            content += f"{prefix}- **{name}/**\n"
            # Liste les fichiers si moins de MAX_FILES_LIST sinon juste mention
            files = [f for f in item.iterdir() if f.is_file() and f.name not in IGNORE_FILES]
            subdirs = [d for d in item.iterdir() if d.is_dir() and d.name not in IGNORE_DIRS]
            if len(files) + len(subdirs) > MAX_FILES_LIST:
                content += f"{prefix}  - ... ({len(files) + len(subdirs)} items)\n"
            else:
                content += walk_dir(item, prefix=prefix + "  ")
        else:
            if name in IGNORE_FILES:
                continue
            content += f"{prefix}- {name}\n"
    return content

def generate_readme(project_dir: Path, output_file: Path = None):
    """Génère un README.md de la structure du projet"""
    if output_file is None:
        output_file = project_dir / "README.md"

    readme_header = f"# {normalize_filename(project_dir.name)}\n\n## Project Organization\n\n```\n"
    readme_footer = "```\n\n### Notes\n\n- Les dossiers volumineux ou contenant de nombreux fichiers sont résumés.\n- Fichiers temporaires ou caches sont ignorés.\n"

    structure = walk_dir(project_dir)
    readme_content = readme_header + structure + readme_footer

    output_file.write_text(readme_content, encoding="utf-8")
    print(f"README généré dans {output_file}")

# Exemple d'utilisation
if __name__ == "__main__":
    project_path = Path(".")  # ton dossier courant
    generate_readme(project_path)
