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
    def __init__(self, model_option, dataset_path, additional_parameters, optimize):
        """
        To do:

        """
        self.model_option = model_option
        self.dataset_path = dataset_path
        self.additional_parameters = additional_parameters
        self.optimize = optimize
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
    
    def preprocess_and_vectorize(self):
        '''
        Processes the input dataframe by filtering, cleaning text data, vectorizing definitions, 
        and creating a new dataframe with features as columns and extracting the target variable.
    '''
        df_filtered = self.dataset[self.dataset['is_obsolete'].isna()] # remove obsolete records
        df_filtered = df_filtered[['id', 'namespace', 'definition']] # removing unecessary columns
        df_filtered['definition'] = df_filtered['definition'].str.replace(r' \[.*?\]$', '', regex=True) # removing text in [] at the end of definitions
        vectorizer = CountVectorizer(stop_words='english', min_df=0.01) # converting definition to feature vectors
        X = vectorizer.fit_transform(df_filtered['definition'])
        dense_X = X.toarray()

        self.X_df = pd.DataFrame(dense_X, columns=vectorizer.get_feature_names_out()) # creating a dataframe for features
        self.y_df = df_filtered['namespace'] # creating df for labels 
        self.vectorizer = vectorizer 
        print('X_df shape:',self.X_df.shape)
        print('y_df shape:',self.y_df.shape)
        print('Data preprocessed and vectorized')

    def transform_input_text(self, input_text):
        """
        Transforms an input text into a feature vector using a pre-fitted CountVectorizer.
        """
        # remocing text in [] (if present)
        cleaned_text = re.sub(r' \[.*?\]$', '', input_text)
        feature_vector = self.vectorizer.transform([cleaned_text])
        input_features = feature_vector.toarray()
        print('Input text transformed')

        return input_features
    
    def initialise_model(self):
        print('Model Initialising')
        if self.model_option == ModelOption.knn:
            self.model = KNNClassifier(self.X_df,self.y_df, k = self.additional_parameters['k'] )
        elif self.model_option == ModelOption.svm:
            self.model = SVMClassifier(self.X_df, self.y_df, self.additional_parameters['C'], self.additional_parameters['kernel'], self.additional_parameters['gamma'])
        elif self.model_option == ModelOption.rf:
            self.model = RFClassifier(self.X_df,self.y_df, self.additional_parameters['n_estimators'], self.additional_parameters['max_depth']
                                      , self.additional_parameters['min_samples_split'], self.additional_parameters['min_samples_leaf']
                                      , self.additional_parameters['bootstrap'])
        print('Model Initialised')

        
    def predict(self, input_text):
        # Transform the input text using the previously fitted vectorizer
        input_features = self.transform_input_text(input_text)

        # Convert the sparse input features to a dense array if necessary
        if isinstance(input_features, csr_matrix):
            input_features_dense = input_features.toarray()
        else:
            input_features_dense = input_features

        # Create a DataFrame with the correct feature names
        input_df = pd.DataFrame(input_features_dense, columns=self.vectorizer.get_feature_names_out())

        # Make the prediction using the DataFrame
        prediction = self.model.predict(input_df)
        return prediction

    def optimize_parameters(self):
        if self.model_option == ModelOption.knn and self.optimize == True:
            k = KNNClassifier.optimize(self)
            self.additional_parameters['k'] == k
        elif self.model_option == ModelOption.svm and self.optimize == True:
            optimized_parameters = SVMClassifier.optimize(self)
            self.additional_parameters.update(optimized_parameters)
        elif self.model_option == ModelOption.rf and self.optimize == True:
            optimized_parameters = RFClassifier.optimize(self)
            self.additional_parameters.update(optimized_parameters)
        
        print('Optimized params=', self.additional_parameters)