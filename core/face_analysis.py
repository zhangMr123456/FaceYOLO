from insightface.app import FaceAnalysis


buffalo_model = FaceAnalysis(name=r"D:\models\insightface\buffalo_l", providers=['CUDAExecutionProvider'])
buffalo_model.prepare(ctx_id=0)


