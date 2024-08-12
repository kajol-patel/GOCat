from gocat_tool.is_a_classifier.is_a_classifier import IsAClassifier, ModelOption
from gocat_tool.is_a_classifier.models.svm import SVMClassifier
from gocat_tool.is_a_classifier.models.random_forest import RFClassifier

DATASET_PATH = "/Users/kajolpatel/Desktop/Individual_Project/GOCat/dataset/go-basic.obo"
#default values
ADDITIONAL_PARAMETERS_MAP = {
    ModelOption.svm: {"C": 5, "kernel": "rbf", "gamma": 'scale'},
    ModelOption.rf: {
        "n_estimators": 220,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "max_depth": 30,
    },
}

if __name__ == "__main__":

    # Predict for some input text
    input_text = "Catalysis of the reaction: 1-palmitoylglycerol-3-phosphate + NADP+ = palmitoylglycerone phosphate + NADPH + H+"
       # "Catalysis of the reaction: calcidiol + H+ + NADPH + O2 = calcitriol + H2O + NADP+",
       # "A heterodimeric complex involved in the release of a nascent polypeptide chain from a ribosome"
    
    optimize = True
    model_option = ModelOption.rf
    no_of_labels = 10

    # Initialize classifier
    classifier = IsAClassifier(
        model_option=model_option,
        dataset_path=DATASET_PATH,
        additional_parameters=ADDITIONAL_PARAMETERS_MAP[model_option],
        optimize=optimize,
        no_of_labels=no_of_labels,
    )
    # dhgvjhs

    prediction = classifier.predict(input_text)

    print("Predicted Namespace:", list(prediction[0]))
