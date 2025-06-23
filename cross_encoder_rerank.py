from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, results, cross_encoder_model=cross_encoder):
    # Grab chunks from dictionary
    documents = [r['document'] for r in results]
    
    # Compute cross encoder relevancy score
    rerank_scores = cross_encoder_model.predict([(query, doc) for doc in documents])
    
    for r, score in zip(results, rerank_scores):
        r['cross_encoder_score'] = float(score)
    
    # Sort results by cross_encoder_score, descending
    results = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)
    
    return results