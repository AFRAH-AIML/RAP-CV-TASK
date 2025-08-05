import numpy as np

class FaceReID:
    def __init__(self, threshold=1.0):
        self.db = {}
        self.next_id = 0
        self.threshold = threshold

    def get_id(self, embedding):
        for id_, db_emb in self.db.items():
            if np.linalg.norm(embedding - db_emb) < self.threshold:
                return id_
        self.db[self.next_id] = embedding
        self.next_id += 1
        return self.next_id - 1