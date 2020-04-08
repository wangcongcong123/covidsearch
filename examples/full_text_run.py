from cord import *

# make sure put the paper collections (four .tar.gz files) and medataset csv file under the dataset_folder
dataset_folder = "../dataset/"
# load metadata and full texts of papers
metadata = load_metadata_papers(dataset_folder, "metadata.csv")
full_papers = load_full_papers(dataset_folder)

# initialize two types of instances for different models that have different computational requirements
# full_input_instances include body text while abstract_input_instances only includes title and abstract
full_input_instances = [(id_, metadata[id_]["title"], metadata[id_]["abstract"], body) for id_, body in
                        full_papers.items() if id_ in metadata]

abstract_input_instances = [(id_, metadata[id_]["title"], metadata[id_]["abstract"]) for id_, body in
                            full_papers.items() if id_ in metadata]

# create two variants of tfidf model that applies full_input_instances and abstract_input_instances respectively
# vectorizer_type can also be count ot bm25
tfidf_full = FullTextModel(full_input_instances, weights=[3, 2, 1], vectorizer_type="tfidf",default_save="../models_save/fulltext")
# the weights argument specifies the importance of each element in the instances list ([1,1] denotes the equal importance to title and abstract in the following case)
tfidf_abstract = FullTextModel(abstract_input_instances, weights=[1, 1], vectorizer_type="tfidf",default_save="../models_save/fulltext")

# here we use tfidf_full as an example to query
query = "covid-19 transmission characteristics"
top_k = 10
start = time.time()
results = tfidf_full.query(query, top_k=top_k)
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

# Query time:  0.3241088390350342
# print results.....top 10
# ==============Query:covid-19 transmission characteristics======================
# Rank # 0 	 score: 0.7518148582260338
# Title:  The epidemiology and pathogenesis of coronavirus disease (COVID-19) outbreak
# Abstract:  Abstract Coronavirus disease (COVID-19) is caused by SARS-COV2 and represents the causative agent of a potentially fatal disease that is of great global public health concern. Based on the large number of infected people that were exposed to the wet animal market in Wuhan City, China, it is suggested that this is likely the zoonotic origin of COVID-19. Person-to-person transmission of COVID-19 infection led to the isolation of patients that were subsequently administered a variety of treatments. Extensive measures to reduce person-to-person transmission of COVID-19 have been implemented to control the current outbreak. Special attention and efforts to protect or reduce transmission should be applied in susceptible populations including children, health care providers, and elderly people. In this review, we highlights the symptoms, epidemiology, transmission, pathogenesis, phylogenetic analysis and future directions to control the spread of this fatal disease.
# Author:  Rothan, Hussin A.; Byrareddy, Siddappa N.
# Journal:  Journal of Autoimmunity
# Publish_time:  2020-02-26
# URL: https://doi.org/10.1016/j.jaut.2020.102433
# ========================================================
# Rank # 1 	 score: 0.7008897231838439
# Title:  Coronavirus Disease 2019 (COVID-19) Pneumonia in a Hemodialysis Patient
# Abstract:  Abstract Coronavirus disease 2019 (COVID-19) is a highly infective disease caused by the severe acute respiratory syndrome coronavirus 2 virus (SARS-CoV-2). Previous studies on COVID-19 pneumonia outbreak were based on information from the general population. Limited data are available for hemodialysis patients with COVID-19 pneumonia. This report describes the clinical characteristics of COVID-19 in an in-center hemodialysis patient as well as our experience in implementing steps to prevent the spread of COVID-19 pneumonia among in-center hemodialysis patients. The diagnosis, infection control, and treatment of COVID-19 in hemodialysis patients are discussed in this report, and we conclude with recommendations for how a dialysis facility can respond to COVID-19 based on our experiences.
# Author:  Tang, Bin; Li, Sijia; Xiong, Yuwan; Tian, Ming; Yu, Jianbin; Xu, Lixia; Zhang, Li; Li, Zhuo; Ma, Jianchao; Wen, Feng; Feng, Zhonglin; Liang, Xinling; Shi, Wei; Liu, Shuangxin
# Journal:  Kidney Medicine
# Publish_time:  2020-03-12
# URL: https://doi.org/10.1016/j.xkme.2020.03.001
# ========================================================
# .......
