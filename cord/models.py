__author__="congcong wang"

import os
import pickle
import shutil
from typing import List

import numpy
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from nltk.corpus import stopwords
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings, FastTextEmbeddings
from flair.data import Sentence
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class BaseModel(object):
    def __init__(self, instances, weights=[1, 1, 1]):
        """
        this is the super-class that specific models need to inherit
        """
        assert len(weights) == len(instances[0][1:])
        self.ids = []
        ids = []
        corpus = []
        for ins in tqdm(instances, desc="Reading instances for vectorization by weights"):
            ids.append(ins[0])
            sequence = ""
            for index, content in enumerate(ins[1:]):
                sequence += (content + " ") * weights[index]
            corpus.append(sequence)
        self.ids = ids
        self.corpus = corpus
        assert len(ids) == len(corpus)

    def query(self, query, top_k=10):
        """
        Search results by query
        :param query: the query used to search results
        :param top_k: the top_k most similar docs are returned
        :return: the search results in the form: [(paper_id,relevance_score),(),...]
        """
        raise NotImplementedError

    def get_similarities(self, query):
        """
        get the similarities between the query and all docs in corpus
        :param query:
        :return: the list of similarities [sim_between_query_and_first_doc,sim_between_query_and_second_doc...]
        """
        raise NotImplementedError

    def save_corpus(self, target_save):
        """
        save the corpus as a sequence of raw text to a the target file
        :param target_save: the path to which the corpus is saved
        """
        with open(target_save, "w", encoding="utf8") as f:
            f.write(" ".join(self.corpus))
        logger.info("corpus is saved to " + target_save)


class FullTextModel(BaseModel):

    def __init__(self, instances, max_features=5000, weights=[1, 1, 1], default_save="models_save/fulltext",
                 vectorizer_type="tfidf"):
        """
        Below is an example of how the arguments should be like
        :param instances: [(id_0, title,abstract,body),(id_1, title,abstract,body),...]
        :param max_features: namely, the length of vector for representing an instance when constructing presentations
        :param weights: [weight for title, weight for abstract, weight for body]
        :param default_save: directory to save pre-trained vectorizer and instances representations of the model
        :param vectorizer_type: the choices of vectorizer include: count, tfidf, bm25
        """
        default_save = default_save + "-" + vectorizer_type + "/"
        super(FullTextModel, self).__init__(instances, weights)
        self.vectorizer_type = vectorizer_type
        if not os.path.isdir(default_save) or not os.path.isfile(
                default_save + "-vectorizer.pkl") or not os.path.isfile(default_save + "-matrix.pkl"):
            if os.path.isdir(default_save):
                shutil.rmtree(default_save)
            os.mkdir(default_save)

            logger.info("Not found the pre-trained model...")
            self.vectorizer = None

            if vectorizer_type == "count":
                self.vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
            elif vectorizer_type == "tfidf" or "bm25":
                self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
            else:
                raise ValueError("Run configuration of vectorizer_type")

            self.X = self.vectorizer.fit_transform(self.corpus)
            with open(default_save + "-vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)
                logger.info("The vectorizer are pre-trained and saved to " + default_save + "-vectorizer.pkl")
            with open(default_save + "-matrix.pkl", 'wb') as f:
                pickle.dump(self.X, f)
                logger.info("Pre-trained matrix are saved to " + default_save + "-matrix.pkl")
        else:
            logger.info("Loading pre-trained vectorizer object from " + default_save)
            with open(default_save + "-vectorizer.pkl", 'rb') as pickled:
                self.vectorizer = pickle.load(pickled)
            logger.info("Loading pre-trained matrix object from " + default_save)
            with open(default_save + "-matrix.pkl", 'rb') as pickled:
                self.X = pickle.load(pickled)
        logger.info("The shape of instances representations are: " + str(self.X.shape))

    def _queryBM25(self, query, b=0.75, k1=1.6):
        """
        Query given a query via BM25 which is actually built upon tf-idf
        :param query: the query to search
        :param b: parameter b in BM25 formula, see: https://en.wikipedia.org/wiki/Okapi_BM25
        :param k1: parameter k1 in BM25 formula, see: https://en.wikipedia.org/wiki/Okapi_BM25
        :return: The array of similarities between the query and all docs
        """
        query_vector = self.vectorizer.transform([query])
        avdl = self.X.sum(1).mean()
        len_X = self.X.sum(1).A1
        assert sparse.isspmatrix_csr(query_vector)
        X = self.X.tocsc()[:, query_vector.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, query_vector.indices] - 1.
        numer = X.multiply(numpy.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

    def get_similarities(self, query):
        """
        get the similarities between the query and all docs in corpus (FullTextModel)
        :param query:
        :return: the list of similarities [sim_between_query_and_first_doc,sim_between_query_and_second_doc...]
        """
        if self.vectorizer_type == "bm25":
            similarities = self._queryBM25(query)
        else:
            query_vector = self.vectorizer.transform([query])
            similarities = linear_kernel(query_vector, self.X).flatten()
        return similarities

    def query(self, query, top_k=10):
        """
        Search results by query (FullTextModel)
        :param query: the query used to search results
        :param top_k: the top_k most similar docs are returned
        :return: the search results in the form: [(paper_id,relevance_score),(),...]
        """
        similarities = self.get_similarities(query)
        related_docs_indices = similarities.argsort()[:-top_k-1:-1]
        return [(self.ids[indice], similarities[indice]) for indice in related_docs_indices]


class WordEmbeddingModel(BaseModel):

    def __init__(self, instances, weights=[1, 1], default_save="models_save/wordembeddingmodel",
                 embedding_list=["glove"], doc_mode="pool", fasttext_path="models_save/fasttext.bin"):
        """
        The embedding functions are supported by the flair(https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md) lib.
        Below is an example of how the arguments should be like
        :param instances: [(id_0, title,abstract),(id_1, title,abstract),...], it is suggestive to exclude body text due to the time and space complexity
        :param weights: [weight for title, weight for abstract, weight for body]
        :param default_save: directory to save the vectorizer and the instances representations of the model
        :param embedding_list: the list of embeddings to concatenate, ["glove"] by default, can also be ["glove","fasttext"] for example. Considering the complexity, it is suggestive to use either glove or fasttext
        :param doc_mode: the model of generating document embedding given word embeddings
        :param fasttext_path: specify the pre-trained fasttext path if fasttext is in embedding_list (for pre-training a fasttext model on CORD19, refer to https://github.com/facebookresearch/fastText#enriching-word-vectors-with-subword-information)
        """
        default_save = default_save + "-" + "".join(embedding_list) + "/"
        super(WordEmbeddingModel, self).__init__(instances, weights)
        if not os.path.isdir(default_save) or not os.path.isfile(
                default_save + "-vectorizer.pkl") or not os.path.isfile(default_save + "-matrix.pkl"):
            if os.path.isdir(default_save):
                shutil.rmtree(default_save)

            os.mkdir(default_save)
            logger.info("Not found the pre-trained model...")
            embeddings = [WordEmbeddings(each) if each == "glove" else FastTextEmbeddings(fasttext_path) for each in
                          embedding_list]
            if doc_mode == "pool":
                # pooling is average by default
                document_embeddings = DocumentPoolEmbeddings(embeddings)
            elif doc_mode == "glove":
                document_embeddings = DocumentRNNEmbeddings(embeddings)
            else:
                raise ValueError("doc_mode should be pool or glove...")
            self.X = []
            for doc in tqdm(self.corpus, desc="Constructing document embeddings..."):
                seq = Sentence(doc)
                document_embeddings.embed(seq)
                self.X.append(seq.get_embedding().tolist())
            self.vectorizer = document_embeddings
            with open(default_save + "-vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)
                logger.info(
                    "Word embedding vectorizer are pre-trained and saved to " + default_save + "-vectorizer.pkl")
            with open(default_save + "-matrix.pkl", 'wb') as f:
                pickle.dump(self.X, f)
                logger.info("Pre-trained Word embedding matrix are saved to " + default_save + "-matrix.pkl")
        else:
            logger.info("Loading pre-trained Word embedding vectorizer object from " + default_save)
            with open(default_save + "-vectorizer.pkl", 'rb') as pickled:
                self.vectorizer = pickle.load(pickled)
            logger.info("Loading pre-trained Word embedding matrix object from " + default_save)
            with open(default_save + "-matrix.pkl", 'rb') as pickled:
                self.X = pickle.load(pickled)
        logger.info("The shape of instances representations are:" + str(len(self.X)) + ", " + str(len(self.X[0])))

    def get_similarities(self, query):
        """
        get the similarities between the query and all docs in corpus (WordEmbeddingModel)
        :param query:
        :return: the list of similarities [sim_between_query_and_first_doc,sim_between_query_and_second_doc...]
        """
        seq = Sentence(query)
        self.vectorizer.embed(seq)
        query_vector = seq.get_embedding().tolist()
        return cosine_similarity([query_vector], self.X).flatten()

    def query(self, query, top_k=10):
        """
        Search results by query (WordEmbeddingModel)
        :param query: the query used to search results
        :param top_k: the top_k most similar docs are returned
        :return: the search results in the form: [(paper_id,relevance_score),(),...]
        """
        cosine_similarities = self.get_similarities(query)
        related_docs_indices = cosine_similarities.argsort()[:-top_k-1:-1]
        return [(self.ids[indice], cosine_similarities[indice]) for indice in related_docs_indices]


class EnsembleModel():
    def __init__(self, model_list: List[BaseModel] = None, weights: List[int] = [1, 1]):
        """
        This class is used to wrap multiple models for docs searching. The combination algorithm implemented in this class so far is just linear combination (TODO: to support more combination methods)
        :param model_list: the list of models to be wrapped
        :param weights: the weights that give importance to each model in the same order as in model_list
        =====
        An example, lets say model_list has model1 and model2, and weights is [1,2] that refers to giving 1/(1+2) importance to model1 and 2/(1+2) to model2
        ====
        """
        assert model_list != None
        self.model_list = model_list
        self.weights = numpy.asarray(weights) / sum(weights)
        self.ids = model_list[0].ids

    def query(self, query, top_k=10):
        """
        the overall work-flow of this method is the same as the query method of each model and the difference here is to combine the similarity scores for final ranking based on the weights.
        (Actually a linear combination is applied here).
        :param query: the query to get search results
        :param top_k: the top_k results are returned
        :return: the search results in the form: [(paper_id,relevance_score),(),...]
        """
        length_check = 0

        similaries = self.model_list[0].get_similarities(query)
        normalised_similarities = (similaries - min(similaries)) / (max(similaries) - min(similaries))
        combined_similarities = normalised_similarities * self.weights[0]

        for index, each_model in enumerate(self.model_list[1:]):
            similaries = self.model_list[0].get_similarities(query)
            normalised_similarities = (similaries - min(similaries)) / (max(similaries) - min(similaries))

            if length_check == 0:
                length_check = normalised_similarities.shape[0]
            assert length_check == normalised_similarities.shape[0]
            combined_similarities += normalised_similarities * self.weights[index + 1]

        related_docs_indices = combined_similarities.argsort()[:-top_k-1:-1]
        return [(self.ids[indice], combined_similarities[indice]) for indice in related_docs_indices]
