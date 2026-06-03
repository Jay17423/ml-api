(ns ml-api.algorithms.count-vectorizer
  "Spark CountVectorizer implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.tokenizer :as tokenizer]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature CountVectorizer]))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row input-field output-field]
  {:words (utils/wrapped-array->clojure (.getAs row input-field))
   :features (str (.getAs row output-field))})

(defn dataset->json
  "Converts dataset into JSON preview."
  [dataset input-field output-field]
  (mapv #(row->clojure % input-field output-field) (.collectAsList dataset)))

(defn fit-transform
  "Fits CountVectorizer model and returns transformed dataset and model."
  [dataset input-field output-field word-limit min-docs min-count]
  (let [count-vectorizer (-> (CountVectorizer.)
                             (.setInputCol input-field)
                             (.setOutputCol output-field)
                             (.setVocabSize word-limit)
                             (.setMinDF min-docs)
                             (.setMinTF min-count))
        model (.fit count-vectorizer dataset)
        transformed-dataset (.transform model dataset)]
    {:dataset transformed-dataset
     :model model}))

(defn transform
  "Transforms tokens into feature vectors."
  [dataset input-field output-field word-limit min-docs min-count]
  (let [tokenized-dataset (tokenizer/transform dataset input-field "words")
        count-vectorizer (-> (CountVectorizer.)
                             (.setInputCol "words")
                             (.setOutputCol output-field)
                             (.setVocabSize word-limit)
                             (.setMinDF min-docs)
                             (.setMinTF min-count))
        model (.fit count-vectorizer tokenized-dataset)]
    (.transform model tokenized-dataset)))

(comment (defn get-vocabulary
  "Returns CountVectorizer vocabulary."
  [dataset input-field word-limit min-docs min-count]
  (let [tokenized-dataset (tokenizer/transform dataset input-field "words")
        count-vectorizer (-> (CountVectorizer.)
                             (.setInputCol "words")
                             (.setOutputCol "features")
                             (.setVocabSize word-limit)
                             (.setMinDF min-docs)
                             (.setMinTF min-count))

        model (.fit count-vectorizer tokenized-dataset)]
    (vec (.vocabulary model)))))

(defn execute
  "Executes Spark CountVectorizer."
  [dataset {:keys [input_field output_field word_limit min_docs min_count]
            :or {output_field "features"
                 word_limit 262144
                 min_docs 1
                 min_count 1.0}}]
  (try
    (log/info {:msg "Starting CountVectorizer"
               :input-field input_field
               :output-field output_field})
    (let [transformed-dataset (transform dataset
                                         input_field
                                         output_field
                                         word_limit
                                         min_docs
                                         min_count)
          preview (dataset->json transformed-dataset "words" output_field)]
      (log/info {:msg "CountVectorizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "CountVectorizer execution failed"
                      {:type :algorithm/count-vectorizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))