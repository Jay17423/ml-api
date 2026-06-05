(ns ml-api.algorithms.tokenizer
  "Spark Tokenizer implementations."
  (:require
   [taoensso.timbre :as log]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature Tokenizer RegexTokenizer]))

(defn transform
  "Transforms text into tokens."
  [ds input-field output-field]
  (let [tokenizer (-> (Tokenizer.)
                      (.setInputCol input-field)
                      (.setOutputCol output-field))]
    (.transform tokenizer ds)))

(defn regex-transform
  "Transforms text using RegexTokenizer."
  [ds input-field output-field pattern pattern-as-delimiter minimum-token-length
   convert-to-lowercase]
  (let [tokenizer (-> (RegexTokenizer.)
                      (.setInputCol input-field)
                      (.setOutputCol output-field)
                      (.setPattern pattern)
                      (.setGaps pattern-as-delimiter)
                      (.setMinTokenLength minimum-token-length)
                      (.setToLowercase convert-to-lowercase))]
    (.transform tokenizer ds)))

(defn execute-tokenizer
  "Executes Spark Tokenizer."
  [ds {:keys [input_field output_field]}]
  (try
    (log/info {:msg "Starting Tokenizer"
               :input-field input_field
               :output-field output_field})
    (let [transformed-ds (transform ds input_field output_field)
          preview (utils/dataset->json transformed-ds
                                       [input_field output_field])]
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
  [ds {:keys [input_field
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
    (let [transformed-ds (regex-transform ds
                                          input_field
                                          output_field
                                          pattern
                                          pattern_as_delimiter
                                          minimum_token_length
                                          convert_to_lowercase)

          preview (utils/dataset->json
                   transformed-ds
                   [input_field output_field])]
      (log/info {:msg "RegexTokenizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "RegexTokenizer execution failed"
                      {:type :algorithm/regex-tokenizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))