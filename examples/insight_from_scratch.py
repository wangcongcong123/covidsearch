from cord import *
# make sure put the paper collections (four .tar.gz files) and medataset csv file under the dataset_folder
dataset_folder = "../dataset/"
# load metadata and full texts of papers
metadata = load_metadata_papers(dataset_folder, "metadata.csv")
full_papers = load_full_papers(dataset_folder)
input_instances = [(id_, metadata[id_]["abstract"] + " " + body) for id_, body in full_papers.items() if id_ in metadata]
sentencesearch = SentenceSearch(input_instances,default_save="../models_save/sentencesearch/")

kaggle_tasks = {"task1": "What is known about transmission, incubation, and environmental stability?",
                "task2": "What do we know about COVID-19 risk factors?",
                "task3": "What do we know about virus genetics, origin, and evolution?",
                "task4": "What do we know about vaccines and therapeutics?",
                "task5": "What do we know about non-pharmaceutical interventions?",
                "task6": "What has been published about medical care?",
                "task7": "Are there geographic variations in the rate of COVID-19 spread? Are there geographic variations in the mortality rate of COVID-19? there any evidence to suggest geographic based virus mutations?",
                "task8": "What do we know about diagnostics and surveillance?",
                "task9": "What has been published about information sharing and inter-sectoral collaboration?",
                "task10": "What has been published about ethical and social science considerations?"}

start = time.time()
sentencesearch.query_by_kaggle_tasks(tasks=kaggle_tasks,top_k=1000)
print("Mining time: ",time.time()-start)