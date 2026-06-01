(ns ml-api.algorithms.ngram
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.tokenizer :as tokenizer])

  (:import
   [org.apache.spark.ml.feature NGram]
   [scala.collection.mutable WrappedArray]))

(defn wrapped-array->clojure 
  [wrapped-array]
  (vec (.array ^WrappedArray wrapped-array)))

(defn row->clojure
  [row output-field]
  {:ngrams (wrapped-array->clojure (.getAs row output-field))})

(defn dataset->json
  [dataset output-field]
  (mapv #(row->clojure % output-field) (.collectAsList dataset)))

(defn execute
  "Executes Spark NGram."

  [dataset {:keys [input_field
                   output_field
                   ngram_size]
            :or {output_field "generated_ngrams"
                 ngram_size 2}}]

  (try
    (log/info {:msg "Starting NGram"
               :input-field input_field
               :output-field output_field
               :ngram-size ngram_size})
    (let [token-column "tokens"
          tokenized-dataset (tokenizer/transform
                             dataset
                             input_field
                             token-column)
          ngram (-> (NGram.)
                    (.setN ngram_size)
                    (.setInputCol token-column)
                    (.setOutputCol output_field))
          transformed-dataset (.transform ngram tokenized-dataset)
          preview (dataset->json transformed-dataset output_field)]

      (log/info
       {:msg "NGram completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "NGram execution failed"
                      {:type :algorithm/ngram-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))