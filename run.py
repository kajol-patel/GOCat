from multigoclassfier_tool.is_a_classifier.is_a_classifier import IsAClassifier, ModelOption as IsAModelOption
from multigoclassfier_tool.namespace_classifier.namespace_classifier import NamespaceClassifier, ModelOption as NamespaceModelOption

#Default settings
DEFAULT_DATASET_PATH = "../dataset/go-basic.obo"

ADDITIONAL_PARAMETERS_MAP = {
    "isa": {
        IsAModelOption.svm: {"C": 5, "kernel": "rbf", "gamma": 'scale'},
        IsAModelOption.rf: {
            "n_estimators": 170,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "bootstrap": True,
            "max_depth": 42
        },
    },
    "namespace": {
        NamespaceModelOption.knn: {"k": 5},
        NamespaceModelOption.svm: {"C": 10, "kernel": "rbf", "gamma": 0.1},
        NamespaceModelOption.rf: {
            "n_estimators": 30,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "bootstrap": True,
            "max_depth": None,
        },
    }
}

def main():
    print("Welcome to the MultiGOClassifier - Gene Ontology Classification tool")
    classifier_type = input("\nPlease choose the classifier type: 'Namespace' classifier or 'Is a' classifier? (namespace/isa): ")
    if classifier_type.lower().startswith("namespace"):
        model_option = input("\nPlease select a model. Options include: K-Nearest Neigbors (KNN), Support Vector Machine (SVM), Random Forest (RF). Enter your choice (KNN/SVM/RF): " )
        model_enum = NamespaceModelOption
    else:
        model_option = input("\nPlease select a model. Options include: Support Vector Machine (SVM), Random Forest (RF). Enter your choice (SVM/RF): ")
        model_enum = IsAModelOption
        no_of_labels_input = input("Please enter the number of top labels to consider OR press enter to proceed with default. ")
        no_of_labels = int(no_of_labels_input) if no_of_labels_input.strip() else 10

    dataset_choice = input("\nWould you like to use a custom dataset or the default dataset? (Type 'custom' OR press enter for default): ")
    if dataset_choice.lower() == 'custom':
        dataset_path = input("Enter the custom dataset path: ")
    if not dataset_choice.strip():
        dataset_path = DEFAULT_DATASET_PATH

    rare_words_exclusion_percent = 1.0
    words_exclusion = input("\nIf you would like to change the rare words exclusion %, enter %, OR Press Enter for default (1%) ):")
    if words_exclusion.strip():
        rare_words_exclusion_percent = float(words_exclusion)

    optimize = input("\nWould you prefer to optimize the hyperparameters or proceed with default settings? (Optimizing may extend processing time. Type 'optimize' or press Enter for default settings): ")
    optimize = optimize.lower() == 'optimize'

    input_text = input("\nPlease provide the text you wish to classify: ")

    model_option = model_enum[model_option.lower()]
    additional_params = ADDITIONAL_PARAMETERS_MAP[classifier_type.lower().split()[0]][model_option]
    
    if classifier_type.lower().startswith("namespace"):
        classifier = NamespaceClassifier(
            model_option=model_option,
            dataset_path=dataset_path,
            additional_parameters=additional_params,
            optimize=optimize,
            rare_words_exclusion_percent = rare_words_exclusion_percent
        )
    else:
        classifier = IsAClassifier(
            model_option=model_option,
            dataset_path=dataset_path,
            additional_parameters=additional_params,
            optimize=optimize,
            no_of_labels=no_of_labels,
            rare_words_exclusion_percent = rare_words_exclusion_percent
        )

    prediction = classifier.predict(input_text)

    if classifier_type.lower().startswith("namespace"):
        if prediction[0] == "biological_process":
            prediction = "Biological Process"
        elif prediction[0] == "molecular_function":
            prediction = "Biological Process"
        elif prediction[0] == "cellular_component":
            prediction = "Cellular Component"
        print("\nPredicted Namespace:", prediction)
    else:
        print("\nPrediction:", ', '.join(prediction[0]))

if __name__ == "__main__":
    main()
