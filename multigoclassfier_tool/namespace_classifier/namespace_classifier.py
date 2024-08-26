from enum import Enum
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from .models.namespace_knn import KNNClassifier
from .models.namespace_svm import SVMClassifier
from .models.namespace_random_forest import RFClassifier
from scipy.sparse import csr_matrix

class ModelOption(Enum):
    knn = "knn"
    svm = "svm"
    rf = "rf"

class NamespaceClassifier():
    def __init__(self, model_option, dataset_path, additional_parameters, optimize, rare_words_exclusion_percent):
        """
        Initialize the NamespaceClassifier with the specified model type and parameters.
        
        :param model_option: The type of model to use (knn, svm, or rf).
        :param dataset_path: The file path to the dataset used for model training.
        :param additional_parameters: Additional parameters specific to the chosen model.
        :param optimize: If True, perform hyperparameter optimization for the model.
        """
        self.model_option = model_option
        self.dataset_path = dataset_path
        self.additional_parameters = additional_parameters
        self.optimize = optimize
        self.rare_words_exclusion_percent = rare_words_exclusion_percent
        self.model = None
        self.parse_obo_file()
        self.preprocess_and_vectorize()
        if self.optimize == True:
            self.optimize_parameters()
        self.initialise_model()
        
    def parse_obo_file(self):
        """        
        Parses an OBO (Open Biomedical Ontology) file to extract terms and create a DataFrame.
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
       #print('Status: Data Parsed')
    
    def preprocess_and_vectorize(self):
        '''
        Processes the input dataframe by filtering, cleaning text data, vectorizing definitions, and creating a new dataframe with features as columns and extracting the target variable.
        '''
        df_filtered = self.dataset[self.dataset['is_obsolete'].isna()] #remove obsolete records
        df_filtered = df_filtered[['id', 'namespace', 'definition']] #removing unecessary columns
        df_filtered['definition'] = df_filtered['definition'].str.replace(r' \[.*?\]$', '', regex=True) #removing text in [] at the end of definitions
        
        vectorizer = CountVectorizer(stop_words='english', min_df=self.rare_words_exclusion_percent/100.0) #converting definition to feature vectors
        X = vectorizer.fit_transform(df_filtered['definition'])
        dense_X = X.toarray()

        self.X_df = pd.DataFrame(dense_X, columns=vectorizer.get_feature_names_out()) # creating a dataframe for features
        self.y_df = df_filtered['namespace'] #creating df for labels 
        self.vectorizer = vectorizer 
        #print('Status: Data preprocessed')

    def transform_input_text(self, input_text):
        """
        Converts input text to a feature vector using the pre-fitted CountVectorizer.

        :param input_text: The text input to transform.
        :return: The transformed feature vector as an array.        
        """
        # removing text in [] (if present)
        cleaned_text = re.sub(r' \[.*?\]$', '', input_text)
        feature_vector = self.vectorizer.transform([cleaned_text])
        input_features = feature_vector.toarray()
        #print('Input text transformed')

        return input_features
    
    def initialise_model(self):
        """
        Initializes the model based on the specified model_option and sets up necessary configurations.
        """
        if self.model_option == ModelOption.knn:
            self.model = KNNClassifier(self.X_df,self.y_df, k = self.additional_parameters['k'] )
        elif self.model_option == ModelOption.svm:
            self.model = SVMClassifier(self.X_df, self.y_df, self.additional_parameters['C'], self.additional_parameters['kernel'], self.additional_parameters['gamma'])
        elif self.model_option == ModelOption.rf:
            self.model = RFClassifier(self.X_df,self.y_df, self.additional_parameters['n_estimators'], self.additional_parameters['max_depth']
                                      , self.additional_parameters['min_samples_split'], self.additional_parameters['min_samples_leaf']
                                      , self.additional_parameters['bootstrap'])
        
    def predict(self, input_text):
        """
        Predicts the namespace based on the input text after transforming and model inference.

        :param input_text: Text input to classify.
        :return The predicted namespace.
        """        
        
        input_features = self.transform_input_text(input_text)
        if isinstance(input_features, csr_matrix):
            input_features_dense = input_features.toarray()
        else:
            input_features_dense = input_features
        input_df = pd.DataFrame(input_features_dense, columns=self.vectorizer.get_feature_names_out())
        prediction = self.model.predict(input_df)

        return prediction

    def optimize_parameters(self):
        """
        Optimizes model parameters based on the model type and the optimization flag set during initialization.
        """

        if self.model_option == ModelOption.knn and self.optimize == True:
            k = KNNClassifier.optimize(self)
            self.additional_parameters['k'] == k
        elif self.model_option == ModelOption.svm and self.optimize == True:
            optimized_parameters = SVMClassifier.optimize(self)
            self.additional_parameters.update(optimized_parameters)
        elif self.model_option == ModelOption.rf and self.optimize == True:
            optimized_parameters = RFClassifier.optimize(self)
            self.additional_parameters.update(optimized_parameters)