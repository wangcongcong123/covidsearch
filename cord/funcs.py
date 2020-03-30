__author__="congcong wang"

import json
import os
import gensim.corpora as corpora
import gensim
from tqdm import tqdm
import pandas as pd
import pickle
import tarfile
import nltk
nltk.download("punkt")
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
stop_words = stopwords.words('english')

def load_full_papers(dataset_folder, full_save="full.pkl"):
    """
    load papers with full body text from subsets of CORD19 dataset
    :param dataset_folder: the directory where the subsets of papers are
    :return:dictionary of pairs (key:paper_id, value:full text body)
    """
    full_save = dataset_folder + full_save
    if not os.path.exists(full_save):
        print("not found", full_save, ", hence load from raw metadata file...")
        full_dirs = [os.path.join(dataset_folder, o) for o in os.listdir(dataset_folder) if o.endswith(".tar.gz")]
        full_papers = {}
        for each in full_dirs:
            tar = tarfile.open(each, "r:gz")
            for member in tqdm(tar.getmembers(), desc="loading tar gz files"):
                f = tar.extractfile(member)
                if f:
                    content = f.read()
                    paper = json.loads(content)
                    full_papers[paper["paper_id"]] = " ".join([obj["text"] for obj in paper["body_text"]])
        with open(full_save, 'wb') as f:
            pickle.dump(full_papers, f)
            print("full papers are saved to " + full_save)
    else:
        print("Loading the full object from", full_save)
        with open(full_save, 'rb') as pickled:
            full_papers = pickle.load(pickled)
    print("loaded:", len(full_papers), "instances")
    return full_papers


def load_metadata_papers(dataset_folder, metadata_file, metadata_save="metadata.pkl"):
    """
    load metadata from metadata.csv
    :param dataset_folder: directory where the metadata is in
    :param metadata_file: raw metadata file name
    :param metadata_save: path to save the loaded metadata as an object
    :return: metadata dictionary of pairs(key:sha,value: other-info)
    """
    metadata = {}
    metadata_path = dataset_folder + metadata_file
    metadata_save = dataset_folder + metadata_save
    if not os.path.exists(metadata_save):
        print("not found", metadata_save, ", hence load from raw metadata file...")
        df = pd.read_csv(metadata_path)
        for index, row in tqdm(df.iterrows(), desc="loading metadata"):
            if row["abstract"] != "Unkown" and not pd.isna(row["abstract"]) and row[
                "abstract"].strip() != "" and not pd.isna(row["sha"]) and not pd.isna(row["title"]):
                metadata[row["sha"]] = {"abstract": row["abstract"], "title": row["title"], "authors": row["authors"],
                                        "journal": row["journal"],
                                        "publish_time": row["publish_time"], "doi": row["doi"]}
        with open(metadata_save, 'wb') as f:
            pickle.dump(metadata, f)
            print("metadata is saved to " + metadata_save)
    else:
        print("Loading the metadata object from", metadata_save)
        with open(metadata_save, 'rb') as pickled:
            metadata = pickle.load(pickled)
    print("loaded:", len(metadata), "instances")
    return metadata

def extract_key_words(corpus, num_topics=5):
    """
    This method is used to extract top 2 keywords from each of [num_topics] topics via LDA (genism) given sentences of a full paper as the input corpus
    This method is implemnted but has still not been used in the live mode system. It is expected to be used in the future version
    :return: key words
    """
    preprocessed = []
    for sent in nltk.sent_tokenize(corpus):
        tokens = simple_preprocess(str(sent))
        ins = [word for word in tokens if word not in stop_words]
        if len(ins) != 0:
            preprocessed.append(ins)

    common_dictionary = corpora.Dictionary(preprocessed)
    id2token = {v: k for k, v in common_dictionary.token2id.items()}

    common_corpus = [common_dictionary.doc2bow(text) for text in preprocessed]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=common_corpus,
                                                passes=10,
                                                random_state=100,
                                                num_topics=num_topics)
    words_selected = []
    for i in range(num_topics):
        words_selected.append([id2token[int(index)] for index, _ in lda_model.show_topic(i)])
    print(words_selected)
    return words_selected

