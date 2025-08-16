import subprocess

subprocess.run([
    "python",
    "recommandation_de_livres/content_dataset.py",
], check=True)

# --- Exécuter le pipeline collaborative ---
subprocess.run([
    "python",
    "recommandation_de_livres/collaborative_dataset.py",
], check=True)

print("✅ All datasets built successfully!")
