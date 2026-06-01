(ns ml-api.algorithms.stop-words-remover
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.tokenizer :as tokenizer])

  (:import
   [org.apache.spark.ml.feature StopWordsRemover]
   [scala.collection.mutable WrappedArray]))


(defn wrapped-array->clojure
  [wrapped-array]
  (vec (.array ^WrappedArray wrapped-array)))

(defn row->clojure
  [row output-field]
  {:filtered_words (wrapped-array->clojure (.getAs row output-field))})

(defn dataset->json
  [dataset output-field]
  (mapv #(row->clojure % output-field) (.collectAsList dataset)))

(defn execute
  "Executes Spark StopWordsRemover."
  [dataset
   {:keys [input_field
           output_field
           stop_words
           case_sensitive]

    :or {output_field "filtered_words"
         case_sensitive false}}]

  (try
    (log/info {:msg "Starting StopWordsRemover"
               :input-field input_field
               :output-field output_field})

    (let [token-column "tokens"
          tokenized-dataset (tokenizer/transform
                             dataset
                             input_field
                             token-column)
          remover (cond->
                   (-> (StopWordsRemover.)
                       (.setInputCol token-column)
                       (.setOutputCol output_field)
                       (.setCaseSensitive case_sensitive))
                    stop_words
                    (.setStopWords
                     (into-array String stop_words)))
          transformed-dataset (.transform remover tokenized-dataset)
          preview (dataset->json transformed-dataset output_field)]

      (log/info {:msg "StopWordsRemover completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "StopWordsRemover execution failed"
                      {:type :algorithm/stop-words-remover-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))