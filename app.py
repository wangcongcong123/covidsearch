from flask_cors import CORS, cross_origin
from flask import Flask, render_template, request
from cord import *
import json

app = Flask(__name__)
CORS(app, supports_credentials=True)
dataset_folder = "dataset/"
metadata = load_metadata_papers(dataset_folder, "metadata.csv")
full_papers = load_full_papers(dataset_folder)

full_input_instances = [(id_, metadata[id_]["title"], metadata[id_]["abstract"], body) for id_, body in
                        full_papers.items() if id_ in metadata]
abstract_input_instances = [(id_, metadata[id_]["title"], metadata[id_]["abstract"]) for id_, body in
                            full_papers.items() if id_ in metadata]

# count = FullTextModel(full_input_instances, weights=[3, 2, 1],default_save="models_save/fulltext",vectorizer_type="count")

# tfidf = FullTextModel(full_input_instances, weights=[3, 2, 1], default_save="models_save/fulltext",
#                       vectorizer_type="tfidf")

# bm25 = FullTextModel(full_input_instances, weights=[3, 2, 1],default_save="models_save/fulltext",vectorizer_type="bm25")
# fasttext = WordEmbeddingModel(abstract_input_instances, weights=[1,1],embedding_list=["fasttext"],default_save="models_save/wordembeddings",doc_mode="pool",fasttext_path="models_save/fasttext.bin")
# bm25_fasttext=EnsembleModel(model_list=[bm25,fasttext],weights=[1,1])
# str2model={"Count":count,"TF-IDF":tfidf,"BM25":bm25,"FastText Embedding":fasttext,"BM25+FastText":bm25_fasttext}

# str2model = {"TF-IDF": tfidf}
str2model={}
insights = SentenceSearch.load_from_save(save_path="models_save/sentencesearch/-results_save.pkl")


def parse_search_results(results):
    returned = []
    for rank, each in enumerate(results):
        parsed = {}
        parsed["title"] = metadata[each[0]]["title"]
        parsed["abstract"] = metadata[each[0]]["abstract"]

        authors = metadata[each[0]]["authors"]
        parsed["authors"] = "None" if pd.isna(authors) else authors
        publish_time = metadata[each[0]]["publish_time"]
        parsed["publish_time"] = "None" if pd.isna(publish_time) else publish_time
        doi = metadata[each[0]]["doi"] if isinstance(metadata[each[0]]["doi"], str) else str(
            metadata[each[0]]["doi"])
        parsed["doi"] = "https://doi.org/" + doi
        returned.append(parsed)
    return returned


def get_insights_results(task_name, top_k=20, second_filter=["covid-19"]):
    top_similar_sentences = insights[task_name]
    top_similar_filtered = []
    for each in top_similar_sentences:
        title_cased = metadata[each[0]]["title"].lower()
        abstract_cased = metadata[each[0]]["abstract"].lower()
        instance = {}
        if any(e in title_cased for e in second_filter) or any(e in abstract_cased for e in second_filter):
            instance["title"] = metadata[each[0]]["title"]
            instance["sentence"] = each[1]
            authors = metadata[each[0]]["authors"]
            instance["authors"] = "None" if pd.isna(authors) else authors
            publish_time = metadata[each[0]]["publish_time"]
            instance["publish_time"] = "None" if pd.isna(publish_time) else publish_time
            doi = metadata[each[0]]["doi"] if isinstance(metadata[each[0]]["doi"], str) else str(
                metadata[each[0]]["doi"])
            instance["doi"] = "https://doi.org/" + doi
            top_similar_filtered.append(instance)
    selected_num = top_k if len(top_similar_filtered) else len(top_similar_filtered)
    return top_similar_filtered[:selected_num]


@app.route('/search', methods=["GET", "POST"])
@cross_origin(supports_credentials=True)
def search():
    query = request.get_json()["query"]
    model_type = request.get_json()["model_type"]
    if model_type not in str2model:
        return json.dumps({"response": "Not Found", "result": "not found this type of model.."})
    results = str2model[model_type].query(query, top_k=20)
    return json.dumps({"response": "Search Results", "result": parse_search_results(results)})
@app.route('/')
def hello_world():
    return render_template('layout.html')


@app.route('/kaggle_task', methods=["GET", "POST"])
@cross_origin(supports_credentials=True)
def getKaggleTaskInsights():
    if request.method == 'GET':
        task_name =request.values.get('task_name')
    else:
        task_name = request.get_json()["task_name"]
    return json.dumps({"response": "Insights Return", "result": get_insights_results(task_name, top_k=100)})

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1")