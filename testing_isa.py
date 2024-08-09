from gocat_tool.is_a_classifier.is_a_classifier import IsAClassifier, ModelOption
from gocat_tool.is_a_classifier.models.svm import SVMClassifier
from gocat_tool.is_a_classifier.models.random_forest import RFClassifier

DATASET_PATH = "/Users/kajolpatel/Desktop/Individual_Project/GOCat/dataset/go-basic.obo"

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
    input_text = "A protein complex that is wholly or partially contained within the lumen or membrane of the extracellular vesicular exosome"
    optimize = False
    model_option = ModelOption.svm

    # Initialize classifier
    classifier = IsAClassifier(
        model_option=model_option,
        dataset_path=DATASET_PATH,
        additional_parameters=ADDITIONAL_PARAMETERS_MAP[model_option],
        optimize=optimize,
    )

    prediction = classifier.predict(input_text)

    print("Predicted Namespace:", prediction)
