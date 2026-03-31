This repository is a Machine Learning classification project on the B3Kat Bibliotheksverbund Bayern Dataset. 

To train a Model, first get the record metadata by applying the function in utils.py or use the preprocessed provided b3kat_new.csv
Train the preprocessor using DataPipeline and choose a model for RandomSearchCV. A pretrained model for each HistGradientBoosting, RandomForest and DecisionTree is provided.
