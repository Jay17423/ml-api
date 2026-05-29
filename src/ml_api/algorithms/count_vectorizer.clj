(ns ml-api.algorithms.count-vectorizer
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.tokenizer :as tokenizer])
  (:import
   [org.apache.spark.ml.feature CountVectorizer]
   [scala.collection.mutable WrappedArray]))

(defn wrapped-array->clojure
  [wrapped-array]
  (vec (.array ^WrappedArray wrapped-array)))

(defn row->clojure
  [row input-field output-field]
  {:words (wrapped-array->clojure (.getAs row input-field))
   :features (str (.getAs row output-field))})

(defn dataset->json
  [dataset input-field output-field]
  (mapv #(row->clojure % input-field output-field) (.collectAsList dataset)))

(defn execute
  [dataset {:keys [input_field
                   output_field
                   word_limit
                   min_docs
                   min_count]

            :or {output_field "features"
                 word_limit 262144
                 min_docs 1
                 min_count 1.0}}]
  (try
    (log/info {:msg "Starting CountVectorizer"
               :input-field input_field
               :output-field output_field})
    
    (let [tokenized-dataset (tokenizer/transform
                             dataset
                             input_field
                             "words")

          count-vectorizer (-> (CountVectorizer.)
                               (.setInputCol "words")
                               (.setOutputCol output_field)
                               (.setVocabSize word_limit)
                               (.setMinDF min_docs)
                               (.setMinTF min_count))
          model (.fit count-vectorizer tokenized-dataset)
          transformed-dataset (.transform model tokenized-dataset)
          preview (dataset->json transformed-dataset "words" output_field)]
      {:vocabulary (vec (.vocabulary model))
       :data preview})

    (catch Exception err
      (throw (ex-info "CountVectorizer execution failed"
                      {:type :algorithm/count-vectorizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))