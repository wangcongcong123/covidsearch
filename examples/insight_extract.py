from cord import *

# make sure put the paper collections (four .tar.gz files) and medataset csv file under the dataset_folder
dataset_folder = "../dataset/"
# load metadata and full texts of papers
metadata = load_metadata_papers(dataset_folder, "metadata.csv")
full_papers = load_full_papers(dataset_folder)

kaggle_tasks = {"task1": "For COVID-19, What is known about transmission, incubation, and environmental stability?",
                "task2": "For COVID-19, What do we know about COVID-19 risk factors?",
                "task3": "For COVID-19, What do we know about virus genetics, origin, and evolution?",
                "task4": "For COVID-19, What do we know about vaccines and therapeutics?",
                "task5": "For COVID-19, What do we know about non-pharmaceutical interventions?",
                "task6": "For COVID-19, What has been published about medical care?",
                "task7": "For COVID-19, Are there geographic variations in the rate of COVID-19 spread? Are there geographic variations in the mortality rate of COVID-19? there any evidence to suggest geographic based virus mutations?",
                "task8": "For COVID-19, What do we know about diagnostics and surveillance?",
                "task9": "For COVID-19, What has been published about information sharing and inter-sectoral collaboration?",
                "task10": "For COVID-19, What has been published about ethical and social science considerations?"}

"""
the file -results_save.pkl is pre-obtained and provided here under models_save/sentencesearch/
Following is the steps to get the insights file from scratch when needed (this may take a while to complete)
 ========== insights mining =====

input_instances = [(id_, metadata[id_]["abstract"] + " " + body) for id_, body in full_papers.items() if id_ in metadata]
sentencesearch = SentenceSearch(input_instances)
start = time.time()
sentencesearch.query_by_kaggle_tasks(tasks=kaggle_tasks,top_k=1000)
print("Mining time: ",time.time()-start)

========== insight mining =====
"""
results = SentenceSearch.load_from_save(save_path="../models_save/sentencesearch/-results_save.pkl")

for task_name, top_similar_sentences in results.items():
    print("==============Sentence Query:" + kaggle_tasks[task_name] + "======================")
    for each in top_similar_sentences:
        title_cased = metadata[each[0]]["title"].lower()
        abstract_cased = metadata[each[0]]["abstract"].lower()
        if "covid-19" in title_cased or "covid-19" in abstract_cased:
            print("-----------------------------------------------------")
            print("Title: ", metadata[each[0]]["title"])
            print("Sentence: ", each[1])
            print("Author: ", metadata[each[0]]["authors"])
            print("Publish_time: ", metadata[each[0]]["publish_time"])
            doi = metadata[each[0]]["doi"] if isinstance(metadata[each[0]]["doi"], str) else str(
                metadata[each[0]]["doi"])
            print("URL: https://doi.org/" + doi)
            print("-----------------------------------------------------")
    print("========================================================")

### The following is the print out
# ==============Sentence Query:What is known about transmission, incubation, and environmental stability?======================
# -----------------------------------------------------
# Title:  The reproductive number R0 of COVID-19 based on estimate of a statistical time delay dynamical system
# Sentence:  The epidemic model for COVID-19 is SEIR if CCDC's data are correct.
# Author:  Nian Shao; Jin Cheng; Wenbin Chen
# Publish_time:  2020-02-20
# URL: https://doi.org/10.1101/2020.02.17.20023747
# -----------------------------------------------------
# -----------------------------------------------------
# Title:  Rational evaluation of various epidemic models based on the COVID-19 data of China
# Sentence:  The SEIR-QD and SEIR-PO models are two suitable ones for modeling COVID-19 by appropriately incorporating the effects of quarantine and self-protection.
# Author:  Wuyue Yang; Dongyan Zhang; Liangrong Peng; Changjing Zhuge; Liu Hong
# Publish_time:  2020-03-16
# URL: https://doi.org/10.1101/2020.03.12.20034595
# -----------------------------------------------------
#.....