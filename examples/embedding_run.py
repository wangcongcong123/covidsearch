from cord import *

# make sure put the paper collections (four .tar.gz files) and medataset csv file under the dataset_folder
dataset_folder = "../dataset/"
# load metadata and full texts of papers
metadata = load_metadata_papers(dataset_folder, "metadata.csv")
full_papers = load_full_papers(dataset_folder)

# here the body texts are excluded due to the complexity of the embedding-based model
abstract_input_instances = [(id_, metadata[id_]["title"], metadata[id_]["abstract"]) for id_, body in
                            full_papers.items() if id_ in metadata]
# create a fasttext model where fasttext_path specifies the path of pre-trained fasttext embeddings
# to pre-train a fasttext on the corpus, refer to: https://github.com/facebookresearch/fastText#enriching-word-vectors-with-subword-information
# to generate a txt corpus, it is suggested to use the following code
'''
full_input_instances = [(id_, metadata[id_]["title"], metadata[id_]["abstract"], body) for id_, body in
                        full_papers.items() if id_ in metadata]
model = FullTextModel(full_input_instances, weights=[1, 1, 1],default_save="../models_save/fulltext",vectorizer_type="count")
model.save_corpus("corpus.txt")
'''
# Here assume there is pre-trained fasttext.bin under models_save folder,
fasttext = WordEmbeddingModel(abstract_input_instances, weights=[1,1],embedding_list=["fasttext"],default_save="../models_save/wordembeddingmodel",fasttext_path="../models_save/fasttext.bin")

# here provides an example to query
query = "covid-19 transmission characteristics"
top_k = 10
start = time.time()
results = fasttext.query(query, top_k=top_k)
print("Query time: ", time.time() - start)
print("print results.....top", top_k)
print("==============Query:" + query + "======================")
for rank, each in enumerate(results):
    print("Rank #", rank, "\t", "score:", each[1])
    print("Title: ", metadata[each[0]]["title"])
    print("Abstract: ", metadata[each[0]]["abstract"])
    print("Author: ", metadata[each[0]]["authors"])
    print("Journal: ", metadata[each[0]]["journal"])
    print("Publish_time: ", metadata[each[0]]["publish_time"])
    print("URL: https://doi.org/" + metadata[each[0]]["doi"])
    print("========================================================")

#### The following is the print out after re-run (the first time runs more time for object serilisation)
# Query time:  2.552046537399292
# print results.....top 10
# ==============Query:covid-19 transmission characteristics======================
# Rank # 0 	 score: 0.8418791261234224
# Title:  Analysis of Spatiotemporal Characteristics of Pandemic SARS Spread in Mainland China
# Abstract:  Severe acute respiratory syndrome (SARS) is one of the most severe emerging infectious diseases of the 21st century so far. SARS caused a pandemic that spread throughout mainland China for 7 months, infecting 5318 persons in 194 administrative regions. Using detailed mainland China epidemiological data, we study spatiotemporal aspects of this person-to-person contagious disease and simulate its spatiotemporal transmission dynamics via the Bayesian Maximum Entropy (BME) method. The BME reveals that SARS outbreaks show autocorrelation within certain spatial and temporal distances. We use BME to fit a theoretical covariance model that has a sine hole spatial component and exponential temporal component and obtain the weights of geographical and temporal autocorrelation factors. Using the covariance model, SARS dynamics were estimated and simulated under the most probable conditions. Our study suggests that SARS transmission varies in its epidemiological characteristics and SARS outbreak distributions exhibit palpable clusters on both spatial and temporal scales. In addition, the BME modelling demonstrates that SARS transmission features are affected by spatial heterogeneity, so we analyze potential causes. This may benefit epidemiological control of pandemic infectious diseases.
# Author:  Cao, Chunxiang; Chen, Wei; Zheng, Sheng; Zhao, Jian; Wang, Jinfeng; Cao, Wuchun
# Journal:  Biomed Res Int
# Publish_time:  2016 Aug 15
# URL: https://doi.org/10.1155/2016/7247983
# ========================================================
# Rank # 1 	 score: 0.8391151856905685
# Title:  Transmission routes of 2019-nCoV and controls in dental practice
# Abstract:  A novel β-coronavirus (2019-nCoV) caused severe and even fetal pneumonia explored in a seafood market of Wuhan city, Hubei province, China, and rapidly spread to other provinces of China and other countries. The 2019-nCoV was different from SARS-CoV, but shared the same host receptor the human angiotensin-converting enzyme 2 (ACE2). The natural host of 2019-nCoV may be the bat Rhinolophus affinis as 2019-nCoV showed 96.2% of whole-genome identity to BatCoV RaTG13. The person-to-person transmission routes of 2019-nCoV included direct transmission, such as cough, sneeze, droplet inhalation transmission, and contact transmission, such as the contact with oral, nasal, and eye mucous membranes. 2019-nCoV can also be transmitted through the saliva, and the fetal–oral routes may also be a potential person-to-person transmission route. The participants in dental practice expose to tremendous risk of 2019-nCoV infection due to the face-to-face communication and the exposure to saliva, blood, and other body fluids, and the handling of sharp instruments. Dental professionals play great roles in preventing the transmission of 2019-nCoV. Here we recommend the infection control measures during dental practice to block the person-to-person transmission routes in dental clinics and hospitals.
# Author:  Peng, Xian; Xu, Xin; Li, Yuqing; Cheng, Lei; Zhou, Xuedong; Ren, Biao
# Journal:  International Journal of Oral Science
# Publish_time:  2020
# URL: https://doi.org/10.1038/s41368-020-0075-9
# ========================================================
#...