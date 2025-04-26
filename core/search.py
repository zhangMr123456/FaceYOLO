from core.database import query_embedding
from core.embedding import get_embeddings_by_media
from core.utils import exception_print
from settings import ALLOWED_IMG_TYPES, ALLOWED_VIDEO_TYPES


@exception_print
def search_function(file_path, top_k, min_score):
    try:
        embs = get_embeddings_by_media(file_path, ALLOWED_IMG_TYPES, ALLOWED_VIDEO_TYPES)
        file_paths = query_embedding(embs, k=top_k)
        results = [(score, file_path) for score, file_path in file_paths if score >= min_score]
    except:
        import traceback
        traceback.print_exc()
        traceback.print_stack()
        raise

    return results
