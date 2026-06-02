(ns ml-api.algorithms.bucketizer
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature Bucketizer]))

(defn row->clojure
  [row feature-fields output-fields]
  (merge (into {} (map (fn [field] [field (.getAs row field)]) feature-fields))
   (into {} (map (fn [field] [field (.getAs row field)]) output-fields))))

(defn dataset->json
  [dataset feature-fields output-fields]

  (mapv #(row->clojure % feature-fields output-fields)
        (.collectAsList dataset)))

(defn splits->2d-array
  [bucket-splits]
  (into-array (map double-array bucket-splits)))


(defn execute
  "Executes Spark Bucketizer."
  [dataset {:keys [feature_field
                   output_field
                   bucket_splits
                   invalid_value]
            :or {invalid_value "error"}}]

  (try
    (log/info {:msg "Starting Bucketizer"
               :feature-field feature_field
               :output-field output_field
               :bucket-splits bucket_splits
               :invalid-value invalid_value})
    (let [bucketizer (-> (Bucketizer.)
                         (.setInputCols (into-array String feature_field))
                         (.setOutputCols (into-array String output_field))
                         (.setSplitsArray (splits->2d-array bucket_splits))
                         (.setHandleInvalid invalid_value))
          transform-dataset (.transform bucketizer dataset)
          preview (dataset->json transform-dataset feature_field output_field)]
      (log/info {:msg "Bucketizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "Bucketizer execution failed"
                      {:type :algorithm/bucketizer-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))