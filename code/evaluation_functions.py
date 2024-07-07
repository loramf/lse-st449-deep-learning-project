
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
import pandas as pd
import numpy as np

def gen_output(G, num_rows, noise_size, one_hot_encoder, conditional=False, site = None, seed=42):
  """
  Generate an output from a trained generator.
  Parameters
  ----------
  G : the trained generator
  num_rows : number of sample rows to be generated
  noise_size : size of the noise vector to be input into the generator
  one_hot_encoder : the one-hot encoder used to decode the output
  conditional : True if generator is conditional
  site : specifies the cancer type that a conditional generator outputs. If None and conditional, cancer types will be
         same as the X_test data
  seed : set the seed for the random noise

  Returns
  -------
  A data sample output by the generator, directly comparable to the X_test data.
  """
  z = tf.random.normal(shape=(num_rows, noise_size), seed=seed)

  if conditional == True and site is None:
    y = tf.convert_to_tensor(ohe.transform(X_test)[:, :116], dtype=tf.float32)
    output = G(z, y, training=False)
  elif conditional == True:
    site_ohe = tf.convert_to_tensor(site_mapping[site], dtype=tf.float32)
    y = tf.reshape(tf.tile(site_ohe, multiples=[num_rows]), shape=(-1, 116))
    output = G(z, y, training=False)
  else:
    output = G(z, training=False)

  return one_hot_encoder.inverse_transform(output)

@ignore_warnings(category=ConvergenceWarning)
def get_accuracy_metrics(X, model_type):
  """
  For a dataset X with n variables, run an ML model to predict each variable from all other variables.
  For each model, X is split into train and test sets, the model fitted on the train and used for prediction on the test.
  The n accuracy scores from prediction are saved in a list.
  Parameters
  ----------
  X : dataset to be evaluated
  model_type : type of ML model - can be logistic regression, 'LR', or random forest, 'RF'

  Returns
  -------
  list of accuracy scores, one for each variable in X
  """
  acc_list = []
  columns = X.columns

  for column in columns:
    y = X[column]
    x = X.drop(column, axis = 1)
    test_ohe = OneHotEncoder(sparse=False)
    test_ohe.fit(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    x_train_ohe = test_ohe.transform(x_train)
    x_test_ohe = test_ohe.transform(x_test)

    if model_type == "RCF":
      model = RandomForestClassifier(n_estimators=10)
    elif model_type == "LR":
      model = LogisticRegression()

    try:
      model.fit(x_train_ohe, y_train)
      pred = model.predict(x_test_ohe)
      acc_list.append(accuracy_score(y_test, pred))
    except:
      acc_list.append(np.nan)
  return acc_list

def compare_accuracy(sample, X_test):
  """
  Get accuracy scores for both generated and real data, calculating the MSE between their accuracy lists.
  Parameters
  ----------
  sample : fake data sample
  X_test : real data sample

  Returns (in order of appearance)
  -------
  MSE for accuracy of logistic regression models
  MSE for accuracy of random forest models
  Accuracy list from logistic regression in sample
  Accuracy list from logistic regression in X_test
  Accuracy list from logistic regression in sample
  Accuracy list from logistic regression in X_test
  """
  lr_sample = get_accuracy_metrics(sample,"LR")
  lr_x_test = get_accuracy_metrics(X_test,"LR")

  #remove any nan occurances
  lr_sample_ = [lr_sample[i] for i in range(len(lr_sample)) if (np.isnan(lr_sample[i]) | np.isnan(lr_x_test[i]))==False]
  lr_x_test_ = [lr_x_test[i] for i in range(len(lr_x_test)) if (np.isnan(lr_sample[i]) | np.isnan(lr_x_test[i]))==False]

  rf_sample = get_accuracy_metrics(sample,"RCF")
  rf_x_test = get_accuracy_metrics(X_test,"RCF")

  return  mean_squared_error(lr_x_test_, lr_sample_), mean_squared_error(rf_x_test, rf_sample), lr_sample_, lr_x_test_, rf_sample, rf_x_test

def probability_lists(sample, X_test, i):
  """
  Get the empirical distributions of a single column in a fake data sample and a real data sample.
  Parameters
  ----------
  sample : fake data sample
  X_test : real data sample
  i : column to get empirical distribution from

  Returns (in order of appearance)
  -------
  empirical distribution of sample
  empirical distribution of X_test
  names of categories in that correspond to distributions
  """
  p_var=sample[i]
  q_var=X_test[X_test.columns[i]]
  p = p_var.value_counts()/len(sample)
  q = q_var.value_counts()/len(X_test)
  pq = pd.DataFrame(q).join(pd.DataFrame(p))
  pq[pq.isna()] = 0
  pq.columns = ['exp', 'obs']
  return pq['obs'].tolist(), pq['exp'].tolist(), pq.index.tolist()

def compare_probs(sample, X_test):
  """
  Compares the empirical distribution of all categorical variables between the fake sample and real sample.
  Calculate the MSE between empirical distributions
  Parameters
  ----------
  sample : fake sample
  X_test : real sample

  Returns
  -------
  MSE between the empirical distributions of sample and X_test sample across all categorical dimensions
  Empirical distributions in sample
  Empirical distributions in X_test
  """
  sample_column_probs = []
  test_column_probs = []

  for i, column in enumerate(X_test.columns):
    sample_list, test_list, _ = probability_lists(sample, X_test, i)
    sample_column_probs += sample_list
    test_column_probs += test_list

  mse = mean_squared_error(sample_column_probs, test_column_probs)

  return mse, sample_column_probs, test_column_probs

def compare_within_site_probs(sample, X_test, conditional = False):
  """
  For each cancer type, compares the empirical distribution of all categorical variables between the fake sample and real sample.
  Calculate the MSE between empirical distributions for each cancer type.
  These are then weighted by the size of the cancer cohort in the real data, X_test and then averaged to get a final score.
  ----------
  sample : fake data sample
  X_test : real data sample
  conditional : True if generator is conditional

  Returns
  -------
  weighted_mse: average weighted MSE for the cancer-specific empirical distributions between the fake and real data .
  mse_list: list of weighted MSE site empirical distributions for each cancer between the fake and real data.
  sample_sizes: list of sizes of fake data sample for each cancer type
  test_sizes: list of sizes of real data sample for each cancer type
  cancer_types: list of cancer types
  """

  position = 0 if conditional == True else 1
  cancer_types = X_test['SITE_ICD10_O2_3CHAR'].unique()
  mse_list = []
  sample_sizes = []
  test_sizes = []
  total_mse = 0
  sample_probs, test_probs, _ = probability_lists(sample, X_test, position)
  for i, site in enumerate(X_test['SITE_ICD10_O2_3CHAR'].unique()):

    sample_site = sample[sample[position]==site]
    test_site = X_test[X_test['SITE_ICD10_O2_3CHAR']==site]
    sample_sizes.append(len(sample_site))
    test_sizes.append(len(test_site))
    mse, _ , _ = compare_probs_new(sample_site, test_site)
    mse_list.append(mse)
    total_mse += mse

  weighted_mse = np.array(total_mse)*np.array(test_sizes)/np.sum(test_sizes)
  return weighted_mse, mse_list, sample_sizes, test_sizes, cancer_types