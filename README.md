# ML API

Apache Spark MLlib REST API built with Clojure.

This project exposes multiple Apache Spark MLlib algorithms through a single unified REST endpoint.

---

# Features

* Apache Spark MLlib integration
* Single REST endpoint for all algorithms
* Clojure-based backend
* JSON request/response structure
* Supports preprocessing, feature engineering, statistics, and ML training algorithms
* Structured logging using Timbre
* Configurable Spark cluster connection

---

# Tech Stack

* Clojure 1.11
* Apache Spark 3.5
* Spark MLlib
* Ring / Compojure
* Timbre Logging
* Omniconf Configuration

---

# Endpoint

```http id="5kjlwm"
POST /v1/ml/execute
```

---

# Generic Request Format

```json id="97xyl4"
{
  "algorithm": "<AlgorithmName>",
  "parameters": {
    "url": "/path/to/data.csv"
  }
}
```

---

# Generic Response Format

```json id="x1x2s9"
{
  "status": "success",
  "algorithm": "<AlgorithmName>",
  "result": {}
}
```

---

# Supported Algorithms

## Statistical Algorithms

* ChiSquareTest

## Feature Extraction

* CountVectorizer

## Feature Transformation

* Tokenizer
* RegexTokenizer
* StopWordsRemover
* NGram
* StringIndexer
* Binarizer
* Normalizer
* StandardScaler
* Bucketizer
* VectorAssembler
* Imputer

## Feature Selection

* ChiSqSelector

## Locality Sensitive Hashing

* BucketedRandomProjectionLSH

## Machine Learning Algorithms

* GeneralizedLinearRegression
* GeneralizedLinearRegressionModel
* LDA

---

# Algorithm API Requests

## 1. ChiSquareTest

```json id="17p6j7"
{
  "algorithm": "ChiSquareTest",
  "parameters": {
    "url": "/data/customer_data.csv",
    "feature_field": ["age","monthly_income","gender"],
    "label_field": "purchase_status",
    "flatten": true
  }
}
```

---

## 2. CountVectorizer

```json id="apv5n0"
{
  "algorithm": "CountVectorizer",
  "parameters": {
    "url": "/data/text_documents.csv",
    "input_field": "words",
    "output_field": "features",
    "word_limit": 5000,
    "min_docs": 2,
    "min_count": 1.0
  }
}
```

---

## 3. Tokenizer

```json id="h0l31w"
{
  "algorithm": "Tokenizer",
  "parameters": {
    "url": "/data/text_data.csv",
    "input_field": "sentence",
    "output_field": "words"
  }
}
```

---

## 4. RegexTokenizer

```json id="ms4cx3"
{
  "algorithm": "RegexTokenizer",
  "parameters": {
    "url": "/data/text_data.csv",
    "input_field": "sentence",
    "output_field": "words",
    "pattern": "\\W",
    "pattern_as_delimiter": true,
    "minimum_token_length": 2,
    "convert_to_lowercase": true
  }
}
```

---

## 5. StopWordsRemover

```json id="f1hnjlwm"
{
  "algorithm": "StopWordsRemover",
  "parameters": {
    "url": "/data/text_data.csv",
    "input_field": "customer_review",
    "output_field": "filtered_review",
    "stop_words": ["a","the","is"],
    "case_sensitive": false
  }
}
```

---

## 6. NGram

```json id="jlwmb4"
{
  "algorithm": "NGram",
  "parameters": {
    "url": "/data/text_data.csv",
    "input_field": "review_text",
    "output_field": "generated_ngrams",
    "ngram_size": 2
  }
}
```

---

## 7. StringIndexer

```json id="7lnexr"
{
  "algorithm": "StringIndexer",
  "parameters": {
    "url": "/data/customer_data.csv",
    "input_field": ["city"],
    "output_field": ["city_index"],
    "invalid_value": "keep",
    "label_order": "frequencyDesc"
  }
}
```

---

## 8. Binarizer

```json id="s3syb5"
{
  "algorithm": "Binarizer",
  "parameters": {
    "url": "/data/student_score.csv",
    "input_field": ["score"],
    "output_field": ["pass_student"],
    "threshold_values": [4.0]
  }
}
```

---

## 9. Normalizer

```json id="e0ik9q"
{
  "algorithm": "Normalizer",
  "parameters": {
    "url": "/data/customer_data.csv",
    "feature_field": ["age","monthly_income","purchase_count"],
    "output_field": "normalized_feature",
    "norm_type": 2
  }
}
```

---

## 10. StandardScaler

```json id="rv7vqg"
{
  "algorithm": "StandardScaler",
  "parameters": {
    "url": "/data/customer_data.csv",
    "feature_field": ["age","monthly_income","purchase_count"],
    "output_field": "scaled_features",
    "with_std": true,
    "with_mean": false
  }
}
```

---

## 11. Bucketizer

```json id="8dfw5e"
{
  "algorithm": "Bucketizer",
  "parameters": {
    "url": "/data/customer_data.csv",
    "feature_field": ["salary"],
    "output_field": ["salary_bucket"],
    "bucket_splits": [[-999999, 30000, 80000, 999999]],
    "invalid_value": "keep"
  }
}
```

---

## 12. VectorAssembler

```json id="rv4dtq"
{
  "algorithm": "VectorAssembler",
  "parameters": {
    "url": "/data/customer_data.csv",
    "feature_field": ["age","monthly_income","purchase_score"],
    "output_field": "features",
    "invalid_value": "keep"
  }
}
```

---

## 13. Imputer

```json id="w5gk6n"
{
  "algorithm": "Imputer",
  "parameters": {
    "url": "/data/customer_data.csv",
    "feature_field": ["salary"],
    "output_field": ["imputed_salary"],
    "strategy": "mean",
    "missing_value": ["NaN"]
  }
}
```

---

## 14. ChiSqSelector

```json id="pxgtq1"
{
  "algorithm": "ChiSqSelector",
  "parameters": {
    "url": "/data/customer_data.csv",
    "feature_field": ["age","salary","purchase_score"],
    "target_field": "purchase_status",
    "output_field": "selected_features",
    "selection_method": "numTopFeatures",
    "top_feature": 2
  }
}
```

---

## 15. BucketedRandomProjectionLSH

```json id="n41fdg"
{
  "algorithm": "BucketedRandomProjectionLSH",
  "parameters": {
    "url": "/data/product.csv",
    "feature_field": ["age","job_score","salary"],
    "output_field": "hashes",
    "bucket_length": 2.0,
    "num_hash_tables": 3,
    "random_seed": 12345
  }
}
```

---

## 16. GeneralizedLinearRegression

```json id="r5k2nx"
{
  "algorithm": "GeneralizedLinearRegression",
  "parameters": {
    "url": "/data/glr_data.csv",
    "feature_field": ["age","salary","purchase_score"],
    "target_field": "purchase_amount",
    "output_field": "predicted_purchase",
    "model_path": "/tmp/glr_model",
    "model_label": "customer_purchase_glr_v1",
    "train_size": 80,
    "seed": 123,
    "regression_family": "gaussian",
    "prediction_link": "identity",
    "include_intercept": true,
    "max_iterations": 25,
    "training_tolerance": 0.000001,
    "regularization_strength": 0.0,
    "training_solver": "irls",
    "aggregation_depth": 2
  }
}
```

---

## 17. GeneralizedLinearRegressionModel

```json id="f9l1xu"
{
  "algorithm": "GeneralizedLinearRegressionModel",
  "parameters": {
    "model_path": "/tmp/glr_model",
    "url": "/data/new_customer_data.csv",
    "feature_field": ["age","salary","purchase_score"],
    "target_field": "purchase_amount",
    "output_field": "predicted_purchase"
  }
}
```

---

## 18. LDA

```json id="j3y20l"
{
  "algorithm": "LDA",
  "parameters": {
    "url": "/data/customer_reviews.csv",
    "text_field": "review_text",
    "topic_count": 10,
    "max_iterations": 20,
    "random_seed": 12345,
    "training_method": "online",
    "checkpoint_interval": 10,
    "initial_learning_offset": 1024,
    "learning_decay_rate": 0.51,
    "training_sample_rate": 0.05,
    "auto_optimize": true,
    "topic_distribution_field": "topic_distribution",
    "keep_last_checkpoint": true
  }
}
```

---

# Error Handling

When an error occurs:

```json id="rqjlwm"
{
  "status": "error",
  "algorithm": "<AlgorithmName>",
  "error": "Human-readable error message"
}
```

## Common Errors

* Unknown algorithm
* Missing required parameter
* File not found
* Invalid datatype
* Missing dataset column
* Spark execution failure
* Invalid model path

---

# Running the Project

## Install Dependencies

```bash id="4s1e0v"
lein deps
```

## Start the Server

```bash id="v1nwnm"
lein run
```

---

# Project Structure

```text id="3c9j5m"
src/
config/
resources/
```

---

# Version

* ML API v1.0
* Apache Spark 3.5
* Clojure 1.11

---

# Author

Jay Prakash
Software Engineer
