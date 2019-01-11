The usage of our code is performing performance evaluation for different entity resolution solutions in different datasets. In order to make it easy to follow and understand, we write all of the evaluation in the following ipython notebook for further study and extension, you can easily run them through ipython/jupyter.

List of notebooks for different dataset for evaluation:
AbtBuyEvaluation.ipynb
AmazonGoogleEvaluation.ipynb
DblpAcmEvaluation.ipynb
DblpScholarEvaluation.ipynb
ItunesAmazonEvaluation.ipynb
WalmartAmazonEvaluation.ipynb

In each of those notebook files, we run totally 7 models for ER in which we have 5 baselines and 2 our approaches. The 5 baselines are average, sif, rnn, attention and hybrid. The 2 our proposed approaches are naive CNN and attention-based CNN. All of the results (training, model selection, and testing) during our experiments are logged into those files, you can check the results there which are the version we reported in our paper. Feel free to re-train and evaluate those models by yourself to see if you can produce similar results. Note that neural network based approaches are sensitive to parameters, you may getter worse or better result by tunning the parameters. The core part of naive CNN and attention-based CNN are also provided embedded in those scripts for the convenience of testing.

Note that all data for training, validation, and testing are also provided and they are already properly split and pre-processed, you don't need to download anything more to run the evaluation we provided.

If you have any questions, feel free to contact us:)
