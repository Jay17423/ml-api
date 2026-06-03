(ns ml-api.algorithms.stop-words-remover
  "Spark StopWordsRemover implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.tokenizer :as tokenizer]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature StopWordsRemover]))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row output-field]
  {:filtered_words (utils/wrapped-array->clojure (.getAs row output-field))})

(defn dataset->json
  "Converts dataset into JSON preview."
  [dataset output-field]
  (mapv #(row->clojure % output-field) (.collectAsList dataset)))

(defn transform
  "Removes stop words from tokens."
  [dataset input-field output-field stop-words case-sensitive]
  (let [token-column "tokens"
        tokenized-dataset (tokenizer/transform dataset input-field token-column)
        remover (cond->
                 (-> (StopWordsRemover.)
                     (.setInputCol token-column)
                     (.setOutputCol output-field)
                     (.setCaseSensitive case-sensitive))
                  stop-words (.setStopWords (into-array String stop-words)))]
    (.transform remover tokenized-dataset)))

(defn execute
  "Executes Spark StopWordsRemover."
  [dataset {:keys [input_field
                   output_field
                   stop_words
                   case_sensitive]
            :or {output_field "filtered_words"
                 case_sensitive false}}]

  (try
    (log/info {:msg "Starting StopWordsRemover"
               :input-field input_field
               :output-field output_field})

    (let [transformed-dataset (transform
                               dataset
                               input_field
                               output_field
                               stop_words
                               case_sensitive)
          preview (dataset->json transformed-dataset output_field)]

      (log/info {:msg "StopWordsRemover completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "StopWordsRemover execution failed"
                      {:type :algorithm/stop-words-remover-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))