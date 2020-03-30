from cord import *

# make sure put the paper collections (four .tar.gz files) and medataset csv file under the dataset_folder
dataset_folder = "dataset/"
# load metadata and full texts of papers
metadata = load_metadata_papers(dataset_folder, "metadata.csv")
full_papers = load_full_papers(dataset_folder)

# initialize two types of instances for different models that have different computational requirements
# full_input_instances include body text while abstract_input_instances only includes title and abstract
full_input_instances=[(id_,metadata[id_]["title"],metadata[id_]["abstract"],body) for id_,body in full_papers.items() if id_ in metadata]
abstract_input_instances = [(id_, metadata[id_]["title"], metadata[id_]["abstract"]) for id_, body in
                            full_papers.items() if id_ in metadata]

bm25 = FullTextModel(full_input_instances, weights=[3, 2, 1],default_save="models_save/fulltext",vectorizer_type="bm25")

fasttext = WordEmbeddingModel(abstract_input_instances, weights=[1, 1], embedding_list=["glove"],
                              fasttext_path="models_save/fasttext.bin")
# combine bm25 and fasttext and give equal importance to each model
ensemble=EnsembleModel(model_list=[bm25,fasttext],weights=[1,1])

# here we use tfidf_full as an example to query
query = "covid-19 transmission characteristics"
top_k = 10
start = time.time()
results = ensemble.query(query, top_k=top_k)
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

# Query time:  0.3650519847869873
# print results.....top 10
# ==============Query:covid-19 transmission characteristics======================
# Rank # 0 	 score: 4.433560368857295
# Title:  Internationally lost COVID-19 cases
# Abstract:  Abstract Background With its epicenter in Wuhan, China, the COVID-19 outbreak was declared a pandemic by the World Health Organization (WHO). While many countries have implemented flight restrictions to China, an increasing number of cases with or without travel background to China are confirmed daily. These developments support concerns on possible unidentified and unreported international COVID-19 cases, which could lead to new local disease epicenters. Methods We have analyzed all available data on the development of international COVID-19 cases from January 20th, 2020 until February 18th, 2020. COVID-19 cases with and without travel history to China were divided into cohorts according to the Healthcare Access and Quality Index (HAQ-Index) of each country. Chi-square and Post-hoc testing were performed. Results While COVID-19 cases with travel history to China seem to peak for each HAQ-cohort, the number of non-travel related COVID-19 cases seem to continuously increase in the HAQ-cohort of countries with higher medical standards. Further analyses demonstrate a significantly lower proportion of reported COVID-19 cases without travel history to China in countries with lower HAQ (HAQ I vs. HAQ II, posthoc p <0.01). Conclusions Our data indicate that countries with lower HAQ-index may either underreport COVID-19 cases or are unable to adequately detect them. Although our data may be incomplete and must be interpreted with caution, inconsistencies in reporting COVID-19 cases is a serious problem which might sabotage efforts to contain the virus.
# Author:  Lau, Hien; Khosrawipour, Veria; Kocbach, Piotr; Mikolajczyk, Agata; Ichii, Hirohito; Schubert, Justyna; Bania, Jacek; Khosrawipour, Tanja
# Journal:  Journal of Microbiology, Immunology and Infection
# Publish_time:  2020-03-14
# URL: https://doi.org/10.1016/j.jmii.2020.03.013
# ========================================================
# Rank # 1 	 score: 4.142147463938454
# Title:  The response of Milan's Emergency Medical System to the COVID-19 outbreak in Italy
# Abstract:  The number of people infected with severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the virus causing coronavirus disease 2019 (COVID-19), is dramatically increasing worldwide.The first person-to-person transmission in Italy was reported on Feb 21, 2020, and led to an infection chain that represents the largest COVID-19 outbreak outside Asia to date. Here we document the response of the Emergency Medical System (EMS) of the metropolitan area of Milan, Italy, to the COVID-19 outbreak.On Jan 30, 2020, WHO declared the COVID-19 outbreak a public health emergency of international concern.2 Since then, the Italian Government has implemented extraordinary measures to restrict viral spread, including interruptions of air traffic from China, organised repatriation flights and quarantines for Italian travellers in China, and strict controls at international airports' arrival terminals. Local medical authorities adopted specific WHO recommendations to identify and isolate suspected cases of COVID-19.Such recommendations were addressed to patients presenting with respiratory symptoms and who had travelled to an endemic area in the previous 14 days or who had worked in the health-care sector, having been in close contact with patients with severe respiratory disease with unknown aetiology. Suspected cases were transferred to preselected hospital facilities where the SARS-CoV-2 test was available and infectious disease units were ready for isolation of confirmed cases.
# Author:  Spina, Stefano; Marrazzo, Francesco; Migliari, Maurizio; Stucchi, Riccardo; Sforza, Alessandra; Fumagalli, Roberto
# Journal:  The Lancet
# Publish_time:  2020-03-20
# URL: https://doi.org/10.1016/S0140-6736(20)30493-1
# ========================================================
#....