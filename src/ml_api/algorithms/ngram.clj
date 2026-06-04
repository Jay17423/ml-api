(ns ml-api.algorithms.ngram
  "Spark NGram implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.tokenizer :as tokenizer]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature NGram]))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row output-field]
  {:ngrams (utils/wrapped-array->clojure (.getAs row output-field))})

(defn dataset->json
  "Converts dataset into JSON preview."
  [ds output-field]
  (mapv #(row->clojure % output-field) (.collectAsList ds)))

(defn transform
  "Transforms tokens into ngrams."
  [ds input-field output-field ngram-size]
  (let [token-column "tokens"
        tokenized-ds (tokenizer/transform ds input-field token-column)
        ngram (-> (NGram.)
                  (.setN ngram-size)
                  (.setInputCol token-column)
                  (.setOutputCol output-field))]
    (.transform ngram tokenized-ds)))

(defn execute
  "Executes Spark NGram."
  [ds {:keys [input_field output_field ngram_size]
       :or {output_field "generated_ngrams" ngram_size 2}}]
  (try
    (log/info {:msg "Starting NGram"
               :input-field input_field
               :output-field output_field
               :ngram-size ngram_size})
    (let [transformed-ds (transform ds input_field output_field ngram_size)
          preview (dataset->json transformed-ds output_field)]
      (log/info {:msg "NGram completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "NGram execution failed"
                      {:type :algorithm/ngram-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))