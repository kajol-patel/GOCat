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
        To do:

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
        data = []
        current_term = {}
        in_term_block = False
        
        with open(self.dataset_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == '[Term]':  #starting a new term block
                    if current_term:
                        data.append(current_term)
                    current_term = {}
                    in_term_block = True
                elif line == '':
                    in_term_block = False  #end of a term block
                elif in_term_block:
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        if key in current_term:  #handling multiple lines of the same key
                            if isinstance(current_term[key], list):
                                current_term[key].append(value)
                            else:
                                current_term[key] = [current_term[key], value]
                        else:
                            current_term[key] = value

        if current_term: #add the last term if file does not end with a newline
            data.append(current_term)

        self.dataset = pd.DataFrame(data)
        if 'def' in self.dataset.columns:
            self.dataset.rename(columns={'def': 'definition'}, inplace=True)
        print('Data Parsed')

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
        Processes and prepares machine learning datasets from the given DataFrame:
        1. Filters out rows where 'is_a' is NaN and selects specific columns.
        2. Cleans 'definition' column and ensures 'is_a' entries are in list format.
        3. Explodes 'is_a' column to separate rows for counting occurrences.
        4. Filters DataFrame to only include rows with top label occurrences.
        5. Transforms 'is_a' column to a multi-label binary format.
        6. Converts 'definition' text to a TF-IDF vector format.

        :param df: Input DataFrame to process.
        :param no_of_labels: Number of top labels to consider for filtering.
        :return: Tuple of DataFrames (X_df, y_df) containing feature vectors and target labels respectively.
        """
        # Step 1: Preprocessing
        df = self.dataset[self.dataset['is_a'].notna()]
        df = df[['id', 'definition', 'is_a']]
        df['is_a'] = df['is_a'].apply(self.extract_go_terms)
        df['definition'] = df['definition'].str.replace(r' \[.*?\]$', '', regex=True)
        df['is_a'] = df['is_a'].apply(lambda x: x if isinstance(x, list) else [x])

        # Step 2: Preparing for machine learning
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
        print('y_df shape:',self.y_df.shape)
        print('X_df shape:',self.X_df.shape)

    def transform_input_text(self, input_texts):
        """
        Transforms an input text into a feature vector using a pre-fitted CountVectorizer.
        """
        
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # removing text in [] (if present)    
        cleaned_texts = [re.sub(r' \[.*?\]', '', text) for text in input_texts]

        # Transform cleaned text into feature vectors
        feature_vectors = self.vectorizer.transform(cleaned_texts)
        input_features = feature_vectors.toarray()
        print('Input features shape:', input_features.shape)
        print('Input text(s) transformed')

        return input_features
    
    def initialise_model(self):

        if self.model_option == ModelOption.svm:
            self.model = SVMClassifier(self.X_df, self.y_df, self.additional_parameters['C'], self.additional_parameters['kernel'], self.additional_parameters['gamma'])
        elif self.model_option == ModelOption.rf:
            self.model = RFClassifier(self.X_df,self.y_df, self.additional_parameters['n_estimators'], self.additional_parameters['max_depth']
                                        , self.additional_parameters['min_samples_split'], self.additional_parameters['min_samples_leaf']
                                        , self.additional_parameters['bootstrap'])
        print('Model Initialised')

    def predict(self, input_text):

        input_features = self.transform_input_text(input_text)

        # Convert the sparse input features to a dense array if necessary
        if isinstance(input_features, csr_matrix):
            input_features_dense = input_features.toarray()
        else:
            input_features_dense = input_features

        # Create a DataFrame with the correct feature names
        input_df = pd.DataFrame(input_features_dense, columns=self.vectorizer.get_feature_names_out())
        # Make the prediction using the DataFrame
        prediction_binary = self.model.predict(input_df)

    # Convert binary predictions to labels
        prediction_labels = self.mlb.inverse_transform(prediction_binary)
        return prediction_labels


    def optimize_parameters(self):

        if self.model_option == ModelOption.svm and self.optimize == True:
            optimized_parameters = SVMClassifier.optimize(self)
            self.additional_parameters['C'] = optimized_parameters['estimator__C']
            self.additional_parameters['kernel'] = optimized_parameters['estimator__kernel']
            self.additional_parameters['gamma'] = optimized_parameters['estimator__gamma']
        elif self.model_option == ModelOption.rf and self.optimize == True:
            optimized_parameters = RFClassifier.optimize(self)
            self.additional_parameters.update(optimized_parameters)
        
        print('Optimized params=', self.additional_parameters)