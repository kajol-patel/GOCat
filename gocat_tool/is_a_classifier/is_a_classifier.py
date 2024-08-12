from enum import Enum
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from .models.svm import SVMClassifier
from .models.random_forest import RFClassifier
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer


class ModelOption(Enum):
    svm = "svm"
    rf = "rf"

class IsAClassifier():

    def __init__(self, model_option, dataset_path, additional_parameters, optimize, no_of_labels):
        """
        Initializes the IsAClassifier with the specified model type, dataset path, additional parameters, and optimization flag.

        :param model_option (ModelOption): The type of model to use (svm or rf).
        :param dataset_path (str): The path to the dataset file.
        :param additional_parameters (dict): A dictionary of additional parameters specific to the chosen model.
        :param optimize (bool): A flag to determine whether to perform hyperparameter optimization.
        :param no_of_labels (int): The number of top labels to consider in multi-label classification.
        """
        self.model_option = model_option
        self.dataset_path = dataset_path
        self.additional_parameters = additional_parameters
        self.optimize = optimize
        self.no_of_labels = no_of_labels
        self.model = None
        self.parse_obo_file()
        self.preprocess_and_vectorize()
        if self.optimize == True:
            self.optimize_parameters()
        self.initialise_model()
    
    def parse_obo_file(self):
        """
        Parses an OBO (Open Biomedical Ontology) formatted file to extract relevant data for classification.
        This method reads the dataset file, extracts terms within '[Term]' blocks, and constructs a DataFrame with these term details.
        """
        data = []
        current_term = {}
        in_term_block = False
        
        with open(self.dataset_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == '[Term]':  
                    if current_term:
                        data.append(current_term)
                    current_term = {}
                    in_term_block = True
                elif line == '':
                    in_term_block = False  
                elif in_term_block:
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        if key in current_term: 
                            if isinstance(current_term[key], list):
                                current_term[key].append(value)
                            else:
                                current_term[key] = [current_term[key], value]
                        else:
                            current_term[key] = value

        if current_term: 
            data.append(current_term)

        self.dataset = pd.DataFrame(data)
        if 'def' in self.dataset.columns:
            self.dataset.rename(columns={'def': 'definition'}, inplace=True)
        #print('Data Parsed')

    def extract_go_terms(self, s):
        go_terms = []

        if isinstance(s, list):
            for item in s:
                go_terms.extend(re.findall(r'GO:\d{7}', item))
        else:
            go_terms = re.findall(r'GO:\d{7}', s)
        return go_terms if len(go_terms) > 1 else (go_terms[0] if go_terms else None)
    

    def preprocess_and_vectorize(self):
        """
        Preprocesses the dataset and vectorizes text data for model training.

        This method filters the dataset for necessary columns and terms, applies transformations to structure it for multi-label classification, and converts definitions to feature vectors.
        """
        
        df = self.dataset[self.dataset['is_a'].notna()]
        df = df[['id', 'definition', 'is_a']]
        df['is_a'] = df['is_a'].apply(self.extract_go_terms)
        df['definition'] = df['definition'].str.replace(r' \[.*?\]$', '', regex=True)
        df['is_a'] = df['is_a'].apply(lambda x: x if isinstance(x, list) else [x])

        exploded_df = df.explode('is_a')
        is_a_of_interest = exploded_df['is_a'].value_counts().head(self.no_of_labels).index.tolist()
        df['is_a'] = df['is_a'].apply(lambda labels: [label for label in labels if label in is_a_of_interest])
        filtered_df = df[df['is_a'].apply(lambda x: any(item in is_a_of_interest for item in x))]
        
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(filtered_df['is_a'])
        self.y_df = pd.DataFrame(y, columns=mlb.classes_)
        
        vectorizer = CountVectorizer(stop_words='english', min_df=0.01)
        X_tfidf = vectorizer.fit_transform(filtered_df['definition'])

        self.X_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
       
        self.vectorizer = vectorizer
        self.mlb = mlb

    def transform_input_text(self, input_texts):
        """
        Transforms an input text into a feature vector using a pre-fitted CountVectorizer.

        :param input_texts: Input text or list of texts to transform.
        :return array: Array of transformed feature vectors.
        """
        
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # removing text in [] (if present)    
        cleaned_texts = [re.sub(r' \[.*?\]', '', text) for text in input_texts]

        feature_vectors = self.vectorizer.transform(cleaned_texts)
        input_features = feature_vectors.toarray()

        return input_features
    
    def initialise_model(self):
        """
        Initializes the classification model based on the specified model_option with the parameters provided.
        """
        if self.model_option == ModelOption.svm:
            self.model = SVMClassifier(self.X_df, self.y_df, self.additional_parameters['C'], self.additional_parameters['kernel'], self.additional_parameters['gamma'])
        elif self.model_option == ModelOption.rf:
            self.model = RFClassifier(self.X_df,self.y_df, self.additional_parameters['n_estimators'], self.additional_parameters['max_depth']
                                        , self.additional_parameters['min_samples_split'], self.additional_parameters['min_samples_leaf']
                                        , self.additional_parameters['bootstrap'])

    def predict(self, input_text):
        """
        Predicts the labels for the provided input text using the trained model.

        :param input_text: The text input for which to predict labels.
        :return list: A list of predicted labels.
        """
        input_features = self.transform_input_text(input_text)

        if isinstance(input_features, csr_matrix):
            input_features_dense = input_features.toarray()
        else:
            input_features_dense = input_features

        input_df = pd.DataFrame(input_features_dense, columns=self.vectorizer.get_feature_names_out())
        prediction_binary = self.model.predict(input_df)

        prediction_labels = self.mlb.inverse_transform(prediction_binary)
        return prediction_labels


    def optimize_parameters(self):
        """
        Performs hyperparameter optimization for the selected model using techniques appropriate to the model's type (SVM or RF).
        Adjusts model parameters to improve performance based on training data.
        """
        if self.model_option == ModelOption.svm and self.optimize == True:
            optimized_parameters = SVMClassifier.optimize(self)
            self.additional_parameters['C'] = optimized_parameters['estimator__C']
            self.additional_parameters['kernel'] = optimized_parameters['estimator__kernel']
            self.additional_parameters['gamma'] = optimized_parameters['estimator__gamma']
        elif self.model_option == ModelOption.rf and self.optimize == True:
            optimized_parameters = RFClassifier.optimize(self)
            self.additional_parameters.update(optimized_parameters)
        #print('Optimized params=', self.additional_parameters)