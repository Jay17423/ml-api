(ns ml-api.algorithms.bucketed-random-projection-lsh
  "Spark Bucketed Random Projection LSH implementation for
  approximate similarity search on feature vectors."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature BucketedRandomProjectionLSH]))

(defn row->clojure
  "Converts Spark hash row into Clojure map."
  [row output-field]
  {:hashes (mapv (fn [hash-vector]
                   (utils/vector->clojure hash-vector))
                 (utils/wrapped-array->clojure (.getAs row output-field)))})

(defn dataset->json
  "Converts transformed dataset into preview JSON."
  [ds output-field]
  (mapv #(row->clojure % output-field) (.collectAsList (.limit ds 20))))

(defn execute
  "Executes Spark BucketedRandomProjectionLSH."
  [ds {:keys [feature_field output_field bucket_length num_hash_tables
              random_seed]
       :or {output_field "hashes" num_hash_tables 1}}]
  (try
    (log/info
     {:msg "Starting BucketedRandomProjectionLSH"
      :feature-field feature_field
      :output-field output_field
      :bucket-length bucket_length
      :num-hash-tables num_hash_tables
      :random-seed random_seed})
    (let [vectorized-ds (va/create-feature-vector ds feature_field)
          lsh (-> (BucketedRandomProjectionLSH.)
                  (.setInputCol "features")
                  (.setOutputCol output_field)
                  (.setBucketLength (double bucket_length))
                  (.setNumHashTables num_hash_tables))
          lsh (if random_seed (.setSeed lsh random_seed)
                  lsh)
          model (.fit lsh vectorized-ds)
          transformed-ds (.transform model vectorized-ds)
          preview (dataset->json transformed-ds output_field)]
      (log/info {:msg "BucketedRandomProjectionLSH completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "BucketedRandomProjectionLSH execution failed"
                      {:type :algorithm/bucketed-random-projection-lsh-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))