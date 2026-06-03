(ns ml-api.algorithms.bucketed-random-projection-lsh
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature BucketedRandomProjectionLSH]))

(defn row->clojure
  [row output-field]

  {:hashes (mapv (fn [hash-vector]
                   (utils/vector->clojure hash-vector))
                 (utils/wrapped-array->clojure
                  (.getAs row output-field)))})

(defn dataset->json
  [dataset output-field]
  (mapv #(row->clojure % output-field) (.collectAsList dataset)))

(defn execute
  "Executes Spark BucketedRandomProjectionLSH."
  [dataset {:keys [feature_field
                   output_field
                   bucket_length
                   num_hash_tables
                   random_seed]
            :or {output_field "hashes"
                 num_hash_tables 1}}]

  (try
    (log/info
     {:msg "Starting BucketedRandomProjectionLSH"
      :feature-field feature_field
      :output-field output_field
      :bucket-length bucket_length
      :num-hash-tables num_hash_tables
      :random-seed random_seed})
    (let [vectorized-dataset (va/create-feature-vector dataset feature_field)
          lsh (-> (BucketedRandomProjectionLSH.)
                  (.setInputCol "features")
                  (.setOutputCol output_field)
                  (.setBucketLength (double bucket_length))
                  (.setNumHashTables num_hash_tables))

          lsh (if random_seed (.setSeed lsh random_seed)
                  lsh)
          model (.fit lsh vectorized-dataset)
          transformed-dataset (.transform model vectorized-dataset)
          preview (dataset->json transformed-dataset output_field)]
      (log/info {:msg "BucketedRandomProjectionLSH completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "BucketedRandomProjectionLSH execution failed"

                      {:type :algorithm/bucketed-random-projection-lsh-failed
                       :feature-field feature_field
                       :error (.getMessage err)}

                      err)))))