from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

class TqdmCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
    def __iter__(self):
        for s in tqdm(self.sentences, desc="Iterating sentences"):
            yield s

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print(f"\n--- Début de l'epoch {self.epoch + 1} ---")
    def on_epoch_end(self, model):
        print(f"--- Fin de l'epoch {self.epoch + 1} ---")
        self.epoch += 1