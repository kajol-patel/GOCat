import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def parse_obo_file(file_path):
    
    data = []
    current_term = {}
    in_term_block = False
    
    with open(file_path, 'r') as file:
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

    return pd.DataFrame(data)

def preprocess_and_vectorize(df):
    '''
    Processes the input dataframe by filtering, cleaning text data, vectorizing definitions, 
    and creating a new dataframe with features as columns and extracting the target variable.
   '''
    
    df_filtered = df[df['is_obsolete'].isna()] # remove obsolete records
    df_filtered = df_filtered[['id', 'namespace', 'definition']] # removing unecessary columns
    df_filtered['definition'] = df_filtered['definition'].str.replace(r' \[.*?\]$', '', regex=True) # removing text in [] at the end of definitions
    vectorizer = CountVectorizer(stop_words='english', min_df=0.01) # converting definition to feature vectors
    X = vectorizer.fit_transform(df_filtered['definition'])
    dense_X = X.toarray()

    X_df = pd.DataFrame(dense_X, columns=vectorizer.get_feature_names_out()) # creating a dataframe for features
    y_df = df_filtered['namespace'] # creating df for labels 

    return X_df, y_df, vectorizer

def transform_input_text(input_text, vectorizer):
    """
    Transforms an input text into a feature vector using a pre-fitted CountVectorizer.
    """
    # remocing text in [] (if present)
    cleaned_text = re.sub(r' \[.*?\]$', '', input_text)
    feature_vector = vectorizer.transform([cleaned_text])
    input_features = feature_vector.toarray()

    return input_features

def train_knn(X_df, y_df, k=5):
    ''' Trains a k-Nearest Neighbors classifier with the provided data and returns the trained model 
    '''
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_df, y_df)
    
    return knn

def train_svm(X_df, y_df, C, kernel, gamma):
    ''' Trains a Support Vector Machine classifier with the provided data and returns the trained model 
    '''
    svm_clf = SVC(C=C, kernel=kernel, gamma=gamma)
    svm_clf.fit(X_df, y_df)
    
    return svm_clf

def train_rf(X_df, y_df, n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap):
    ''' Trains a Random Forest classifier with the provided data and returns the trained model 
    '''
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap )
    rf_clf.fit(X_df, y_df)
    
    return rf_clf

def get_model(X_df, y_df, model_type, extra_parameters):

    if model_type == 'KNN':
        model = train_knn(X_df,y_df, extra_parameters)
    elif model_type == 'SVM':
        model = train_svm(X_df,y_df, extra_parameters)
    elif model_type == 'RF':
        model = train_rf(X_df,y_df, extra_parameters)

    print('model initialised')
    return model

def predict(model, input_features):
    prediction = model.predict(input_features)
    print('value predicted')
    return prediction

model = get_model(None, None)
prediction = predict(None,None)

