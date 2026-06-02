(ns ml-api.algorithms.tokenizer
  "Spark Tokenizer implementations."
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature Tokenizer RegexTokenizer]
   [scala.collection.mutable WrappedArray]))

(defn wrapped-array->clojure
  "Converts WrappedArray into Clojure vector."
  [wrapped-array]
  (vec (.array ^WrappedArray wrapped-array)))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row input-field output-field]
  {:input (.getAs row input-field)
   :tokens (wrapped-array->clojure (.getAs row output-field))})

(defn dataset->json
  "Converts dataset into JSON preview."
  [dataset input-field output-field]
  (mapv #(row->clojure % input-field output-field) (.collectAsList dataset)))

(defn transform
  "Transforms text into tokens."
  [dataset input-field output-field]
  (let [tokenizer (-> (Tokenizer.)
                      (.setInputCol input-field)
                      (.setOutputCol output-field))]
    (.transform tokenizer dataset)))

(defn regex-transform
  "Transforms text using RegexTokenizer."
  [dataset input-field output-field pattern pattern-as-delimiter 
   minimum-token-length convert-to-lowercase]
  (let [tokenizer (-> (RegexTokenizer.)
                      (.setInputCol input-field)
                      (.setOutputCol output-field)
                      (.setPattern pattern)
                      (.setGaps pattern-as-delimiter)
                      (.setMinTokenLength minimum-token-length)
                      (.setToLowercase convert-to-lowercase))]
    (.transform tokenizer dataset)))

(defn execute-tokenizer
  "Executes Spark Tokenizer."
  [dataset {:keys [input_field output_field]}]

  (try
    (log/info {:msg "Starting Tokenizer"
               :input-field input_field
               :output-field output_field})
    (let [transformed-dataset (transform dataset input_field output_field)
          preview (dataset->json transformed-dataset input_field output_field)]
      (log/info {:msg "Tokenizer completed successfully"})
      {:data preview})

    (catch Exception err
      (throw (ex-info "Tokenizer execution failed"
                      {:type :algorithm/tokenizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))

(defn execute-regex-tokenizer
  "Executes Spark RegexTokenizer."
  [dataset {:keys [input_field
                   output_field
                   pattern
                   pattern_as_delimiter
                   minimum_token_length
                   convert_to_lowercase]

            :or {pattern "\\s+"
                 pattern_as_delimiter true
                 minimum_token_length 1
                 convert_to_lowercase true}}]

  (try
    (log/info {:msg "Starting RegexTokenizer"
               :input-field input_field
               :output-field output_field})

    (let [transformed-dataset (regex-transform
                               dataset 
                               input_field
                               output_field
                               pattern
                               pattern_as_delimiter
                               minimum_token_length
                               convert_to_lowercase)
          preview (dataset->json transformed-dataset input_field output_field)]

      (log/info {:msg "RegexTokenizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "RegexTokenizer execution failed"
                      {:type :algorithm/regex-tokenizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))