(ns ml-api.algorithms.tokenizer
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature Tokenizer RegexTokenizer]
   [scala.collection.mutable WrappedArray]))

(defn wrapped-array->clojure
  [wrapped-array]
  (vec (.array ^WrappedArray wrapped-array)))

(defn row->clojure
  [row input-field output-field]
  {:input (.getAs row input-field)
   :tokens (wrapped-array->clojure (.getAs row output-field))})

(defn dataset->json
  [dataset input-field output-field]
  (mapv #(row->clojure % input-field output-field) (.collectAsList dataset)))

(defn transform
  [dataset input-field output-field]

  (let [tokenizer
        (-> (Tokenizer.)
            (.setInputCol input-field)
            (.setOutputCol output-field))]
    (.transform tokenizer dataset)))

(defn execute-tokenizer
  [dataset {:keys [input_field output_field]}]

  (try
    (log/info {:msg "Starting Tokenizer"
               :input-field input_field
               :output-field output_field})
    (let [tokenizer (-> (Tokenizer.)
                        (.setInputCol input_field)
                        (.setOutputCol output_field))
          transformed-dataset (.transform tokenizer dataset)
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
    (let [tokenizer (-> (RegexTokenizer.)
                        (.setInputCol input_field)
                        (.setOutputCol output_field)
                        (.setPattern pattern)
                        (.setGaps pattern_as_delimiter)
                        (.setMinTokenLength minimum_token_length)
                        (.setToLowercase convert_to_lowercase))
          transformed-dataset (.transform tokenizer dataset)
          preview (dataset->json transformed-dataset input_field output_field)]
      (log/info {:msg "RegexTokenizer completed successfully"})
      {:data preview}) 
    (catch Exception err
      (throw (ex-info "RegexTokenizer execution failed"
                      {:type :algorithm/regex-tokenizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))