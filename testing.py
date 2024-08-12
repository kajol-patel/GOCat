from gocat_tool.namespace_classifier.namespace_classifier import NamespaceClassifier, ModelOption

DATASET_PATH = "/Users/kajolpatel/Desktop/Individual_Project/GOCat/dataset/go-basic.obo"

ADDITIONAL_PARAMETERS_MAP = {
    ModelOption.knn: {"k": 5},
    ModelOption.svm: {"C": 10, "kernel": "rbf", "gamma": 0.1},
    ModelOption.rf: {
        "n_estimators": 200,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "bootstrap": False,
        "max_depth": None,
    },
}

if __name__ == "__main__":

    # Predict for some input text
    input_text = "A protein complex that is wholly or partially contained within the lumen or membrane of the extracellular vesicular exosome"
    optimize = False
    model_option = ModelOption.svm

    # Initialize classifier
    classifier = NamespaceClassifier(
        model_option=model_option,
        dataset_path=DATASET_PATH,
        additional_parameters=ADDITIONAL_PARAMETERS_MAP[model_option],
        optimize=optimize,
    )

    prediction = classifier.predict(input_text)
    if prediction[0] == "biological_process":
        prediction = "Biological Process"
    elif prediction[0] == "molecular_function":
        prediction = "Biological Process"
    elif prediction[0] == "cellular_component":
        prediction = "Cellular Component"

    print("Predicted Namespace:", prediction)
