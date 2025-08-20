import subprocess

subprocess.run([
    "python",
    "recommandation_de_livres/content_dataset_depository.py",
], check=True)

subprocess.run([
    "python",
    "recommandation_de_livres/collaborative_dataset.py",
], check=True)

print("All datasets built successfully !")