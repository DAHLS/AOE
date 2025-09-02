import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import argparse
import warnings
import datetime
import pickle
warnings.filterwarnings('ignore')

# Algorithm configuration dictionary
ALGORITHM_CONFIGS = {
    'svm': {
        'class': SVC,
        'params': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
        'train_func': lambda model, X_train, y_train: model.fit(X_train, y_train),
        'predict_func': lambda model, X_test: model.predict(X_test)
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'train_func': lambda model, X_train, y_train: model.fit(X_train, y_train),
        'predict_func': lambda model, X_test: model.predict(X_test)
    },
    'logistic_regression': {
        'class': LogisticRegression,
        'params': {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']},
        'train_func': lambda model, X_train, y_train: model.fit(X_train, y_train),
        'predict_func': lambda model, X_test: model.predict(X_test)
    },
    'kNN': {
        'class': KNeighborsClassifier,
        'params': {'n_neighbors': [3, 5, 7, 9],
                   'weights': ['uniform', 'distance']},
        'train_func': lambda model, X_train, y_train: model.fit(X_train, y_train),
        'predict_func': lambda model, X_test: model.predict(X_test)
    },
    'decision_tree': {
        'class': DecisionTreeClassifier,
        'params': {'max_depth': [3, 5, 10, None],
                   'min_samples_split': [2, 5, 10]},
        'train_func': lambda model, X_train, y_train: model.fit(X_train, y_train),
        'predict_func': lambda model, X_test: model.predict(X_test)
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': [50, 100, 200],
                   'learning_rate': [0.1, 0.05, 0.01],
                   'max_depth': [3, 5, 7]},
        'train_func': lambda model, X_train, y_train: model.fit(X_train, y_train),
        'predict_func': lambda model, X_test: model.predict(X_test)
    },
    'naive_bayes': {
        'class': GaussianNB,  # This is from sklearn.naive_bayes
        'params': {'var_smoothing': [1e-9, 1e-8, 1e-7]},
        'train_func': lambda model, X_train, y_train: model.fit(X_train, y_train),
        'predict_func': lambda model, X_test: model.predict(X_test)
    }
}

def load_and_preprocess_data():
    # This is a placeholder - replace with your actual data loading logic
    pure_df = pd.read_excel('data/Org_dump_processed.xlsx')
    texts = pure_df['Text']
    labels = pure_df['Orgs_parents']
    #[1, 1, 0, 0]  # 1 for positive, 0 for negative
    return texts, labels

def create_bow_features(texts, vectorizer_type='count', 
                        max_features=None, ngrams=2):
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english',
                                     ngram_range=(1,ngrams))
    else:
        vectorizer = CountVectorizer(max_features=max_features, stop_words='english',
                                     ngram_range=(1,ngrams))  
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def tune_hyperparameters(X_train, y_train, algorithm_config, algorithm_name):
    """Perform hyperparameter tuning using GridSearchCV"""
    print(f"Starting hyperparameter tuning for {algorithm_name}...")
    
    model_class = algorithm_config['class']
    param_grid = algorithm_config['params']
    
    try:
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            model_class(), 
            param_grid, 
            cv=3,  # Use 3-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {algorithm_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Return the best model
        return grid_search.best_estimator_
        
    except Exception as e:
        print(f"Error during hyperparameter tuning for {algorithm_name}: {e}")
        return None
    

def evaluate_model(model, X_test, y_test, predict_func):
    predictions = predict_func(model, X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')   
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
    }

def save_model(model, filename):
    """Save a trained model to pickle file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open('models/' + f'{timestamp}' + filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def run_algorithm_comparison(X_train, X_test, y_train, y_test, 
                             algorithms_config, save_model='n', hyperp='n'):
    results = {}
    for algo_name, config in algorithms_config.items():
        print("Testing " + algo_name + "...")      
        # Create model instance
        try:
            if hyperp == 'y':
                print(f"Performing hyperparameter tuning for {algo_name}...")
                tuned_model = tune_hyperparameters(X_train, y_train, config, algo_name)
                if tuned_model is None:
                    print(f"Skipping {algo_name} due to tuning error")
                    continue
                model = tuned_model
            else:
                model = config['class']()
        except Exception as e:
            print("Error creating " + algo_name + ": " + str(e))
            continue        
        # Train model
        try:
            config['train_func'](model, X_train, y_train)
        except Exception as e:
            print("Error training " + algo_name + ": " + str(e))
            continue     
        # Predict and evaluate
        try:
            evaluation = evaluate_model(model, X_test, y_test, config['predict_func'])
            if evaluation:
                results[algo_name] = {
                    'model': model,
                    'accuracy': evaluation['accuracy'],
                    'precision': evaluation['precision'],
                    'recall': evaluation['recall'],
                    'f1': evaluation['f1'],
                    'predictions': evaluation['predictions'],
                }
                print(f"{algo_name} Report: \n" + 
                      str(classification_report(y_test, evaluation['predictions'])))
                if save_model == 'y':
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"{algo_name}_model_{timestamp}.pkl"
                    save_model(model, model_filename)
        except Exception as e:
            print("Error evaluating " + algo_name + ": " + str(e))
            continue
    return results

def compare_results(results_dict):
    df_results = []   
    for algo_name, result in results_dict.items():
        df_results.append({
            'Algorithm': algo_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1']
        })
    df = pd.DataFrame(df_results)
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*60)
    print(df.sort_values('F1-Score', ascending=False))
    return df

def save_results(results, filename='algorithm_comparison.csv'):
    df = compare_results(results)
    df.to_csv(filename, index=False)
    print("\nResults saved to " + filename)


def main_with_args():
    parser = argparse.ArgumentParser(description='Compare ML algorithms on AOE data')
    parser.add_argument('--algo', nargs='+', 
                       choices=['svm', 'random_forest', 
                                'logistic_regression', 'kNN', 'all'],
                       default=['all'], help='Algorithms to test')
    parser.add_argument('--feat', nargs='?', default=None, type=int,
                        help='Model size')
    parser.add_argument('--savemodel', nargs='?', choices=['y', 'n'],
                        default=['n'], help='Save models to file')
    parser.add_argument('--hyperpara', nargs='?', choices=['y', 'n'],
                        default=['n'], help='Enable hyperparameter tuning')
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    try:
        texts, labels = load_and_preprocess_data()
        
        # Create bag-of-words features
        y = labels.values
        X, vectorizer = create_bow_features(texts, vectorizer_type='tfidf',
                                            max_features=args.feat)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=y)
        print("Training set size: " + str(X_train.shape))
        print("Test set size: " + str(X_test.shape))
        
    except Exception as e:
        print("Error in data loading/preprocessing: " + str(e))
        return
    
    # Determine which algorithms to run
    if 'all' in args.algo:
        configs = ALGORITHM_CONFIGS
    else:
        configs = {name: ALGORITHM_CONFIGS[name] for name in args.algo 
                   if name in ALGORITHM_CONFIGS}
    
    print("Testing algorithms: " + str(list(configs.keys())))
    
    # Run algorithm comparison
    results = run_algorithm_comparison(X_train, X_test, y_train, y_test, configs,
                                       save_model=args.savemodel, hyperp=args.hyperpara)
    
    if results and len(results) > 1:
        save_results(results)
    
    
if __name__ == "__main__":
    main_with_args()