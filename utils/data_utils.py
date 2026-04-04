from utils.tfidf import TfIdf

def choose_top_idx_tfidf(knowledge, question):
    table = TfIdf()
    for i, paragraph in enumerate(knowledge):
        table.add_document(i, paragraph)
    results = table.similarities(question)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[0]