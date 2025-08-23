import subprocess

subprocess.run([
    "python",
    "recommandation_de_livres/fusion_content_dataset.py",
], check=True)

subprocess.run([
    "python",
    "recommandation_de_livres/fusion_collaborative_dataset.py",
], check=True)



print("All datasets built successfully !")