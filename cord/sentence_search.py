__author__="congcong wang"

import pickle
import shutil
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class SentenceSearch():
    def __init__(self,instances,default_save="models_save/sentencesearch/",sentence_transformer='bert-base-nli-mean-tokens'):
        """
        This class is used to extract insights as specified in https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
        Considering the complexity of this method, this model is considered to be inapproporiate for query-based search
        :param instances: [(paper-id,paper-content),...]
        :param default_save: path for object serialization, we need this due to both large space and time computation complexity from scratch every time
        :param sentence_transformer: "bert-base-nli-mean-tokens" by default, more refers to https://github.com/UKPLab/sentence-transformers. This is expected to adapt to in-domain pre-trained models such as SciBERT or BioBERT
        """
        self.index2idsent = {}
        self.embeddings=[]
        self.default_save=default_save
        self.embedder = SentenceTransformer(sentence_transformer)
        if not os.path.isdir(default_save) or not os.path.isfile(
                default_save + "-embeddings.pkl") or not os.path.isfile(default_save + "-index2idsent.pkl"):

            if os.path.isdir(default_save):
                shutil.rmtree(default_save)
            os.mkdir(default_save)
            logger.info("Not found the pre-saved files...")

            sentences_batch = []
            batch_size = 8
            index=0
            for ins in tqdm(instances, desc="Reading sentences from instances"):
                for sent in nltk.sent_tokenize(ins[1]):
                    if len(sent)>=15:
                        self.index2idsent[index]=(ins[0],sent)
                        index+=1
                        sentences_batch.append(sent)
                    if index%batch_size==0:
                        batch_embeddings=self.embedder.encode(sentences_batch)
                        self.embeddings.extend(batch_embeddings)
                        sentences_batch=[]
            if sentences_batch!=[]:
                batch_embeddings = self.embedder.encode(sentences_batch)
                self.embeddings.extend(batch_embeddings)
            assert len(self.embeddings)==len(self.index2idsent)
            with open(default_save+"-embeddings.pkl", 'wb') as f:
                pickle.dump(self.embeddings, f)
                logger.info("embeddings are saved to " + default_save+"-embeddings.pkl")
            with open(default_save+"-index2idsent.pkl", 'wb') as f:
                pickle.dump(self.index2idsent, f)
                logger.info("Index2idsent is saved to " + default_save+"-index2idsent.pkl")
        else:
            logger.info("Loading sentences embeddings object from " + default_save + "-embeddings.pkl")
            with open(default_save + "-embeddings.pkl", 'rb') as pickled:
                self.embeddings = pickle.load(pickled)
            logger.info("Loading ids object from " + default_save + "-index2idsent.pkl")
            with open(default_save + "-index2idsent.pkl", 'rb') as pickled:
                self.index2idsent = pickle.load(pickled)
        logger.info("Shape of embeddings: "+str(len(self.embeddings))+","+str(len(self.embeddings[0])))

    def query_by_kaggle_tasks(self,tasks,top_k=100):
        """
        This method is used to query insights for each task as in https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
        :param tasks: the dictionary paired by {"task_name":"task_desc",...}
        :param top_k: select the top k most semantic similar sentences from the sentence-level corpus, namely the insights for each task
        :return: {"task_name",[top_k_sentences],}
        """
        tasks2ranked_indices={}
        if not os.path.isdir(self.default_save) or not os.path.isfile(
                self.default_save + "-tasks2ranked_indices.pkl"):
            for name,query in tqdm(tasks.items()):
                logger.info("Computing for "+name)
                query_embedding = self.embedder.encode([query])
                start = time.time()
                cosine_similarities=cosine_similarity(query_embedding,self.embeddings).flatten()
                ranked_indices=cosine_similarities.argsort()
                tasks2ranked_indices[name]=ranked_indices
                logger.info(("Test time: "+str(time.time() - start)))
            with open(self.default_save+"-tasks2ranked_indices.pkl", 'wb') as f:
                pickle.dump(tasks2ranked_indices, f)
                logger.info("tasks2ranked_indices is saved to " + self.default_save+"-tasks2ranked_indices.pkl")
        else:
            logger.info("Loading tasks2ranked_indices object from " + self.default_save + "-tasks2ranked_indices.pkl")
            with open(self.default_save + "-tasks2ranked_indices.pkl", 'rb') as pickled:
                tasks2ranked_indices = pickle.load(pickled)
        return_results={}
        for task_name,ranked_indices in tasks2ranked_indices.items():
            related_sents_indices = ranked_indices[:-top_k-1:-1]
            results=[(self.index2idsent[indice][0],self.index2idsent[indice][1]) for indice in related_sents_indices]
            return_results[task_name]=results
        with open(self.default_save + "-results_save.pkl", 'wb') as f:
            pickle.dump(return_results, f)
            logger.info("results are saved to " + self.default_save + "-results_save.pkl")
        return return_results

    @classmethod
    def load_from_save(self,save_path="models_save/sentencesearch/-results_save.pkl"):
        with open(save_path, 'rb') as pickled:
            return_results = pickle.load(pickled)
        return return_results
