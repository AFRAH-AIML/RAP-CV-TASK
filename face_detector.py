from insightface.app import FaceAnalysis

def load_face_detector():
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # GPU enabled
    app.prepare(ctx_id=0)
    return app