(ns ml-api.algorithms.stop-words-remover
  "Spark StopWordsRemover implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.tokenizer :as tokenizer]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature StopWordsRemover]))

(defn transform
  "Removes stop words from tokens."
  [ds input-field output-field stop-words case-sensitive]
  (let [token-column "tokens"
        tokenized-ds (tokenizer/transform ds input-field token-column)
        remover (cond->
                 (-> (StopWordsRemover.)
                     (.setInputCol token-column)
                     (.setOutputCol output-field)
                     (.setCaseSensitive case-sensitive))
                  stop-words (.setStopWords (into-array String stop-words)))]
    (.transform remover tokenized-ds)))

(defn execute
  "Executes Spark StopWordsRemover."
  [ds {:keys [input_field output_field stop_words case_sensitive]
       :or {output_field "filtered_words" case_sensitive false}}]
  (try
    (log/info {:msg "Starting StopWordsRemover"
               :input-field input_field
               :output-field output_field})
    (let [transformed-ds (transform ds input_field output_field stop_words
                                    case_sensitive)
          preview (utils/dataset->json transformed-ds [output_field])]
      (log/info {:msg "StopWordsRemover completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "StopWordsRemover execution failed"
                      {:type :algorithm/stop-words-remover-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))