(ns ml-api.algorithms.bucketizer
  "Spark Bucketizer implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature Bucketizer]))

(defn splits->2d-array
  "Converts bucket splits into 2D double array."
  [bucket-splits]
  (into-array (map double-array bucket-splits)))

(defn transform
  "Transforms numeric values into buckets."
  [ds feature-fields output-fields bucket-splits invalid-value]
  (let [bucketizer (-> (Bucketizer.)
                       (.setInputCols (into-array String feature-fields))
                       (.setOutputCols (into-array String output-fields))
                       (.setSplitsArray (splits->2d-array bucket-splits))
                       (.setHandleInvalid invalid-value))]
    (.transform bucketizer ds)))

(defn execute
  "Executes Spark Bucketizer."
  [ds {:keys [feature_field output_field bucket_splits invalid_value]
       :or {invalid_value "error"}}]
  (try (log/info {:msg "Starting Bucketizer"
                  :feature-field feature_field
                  :output-field output_field
                  :bucket-splits bucket_splits
                  :invalid-value invalid_value})
       (let [transformed-ds (transform ds feature_field output_field
                                       bucket_splits
                                       invalid_value)
             preview (utils/dataset->json
                      transformed-ds
                      (utils/preview-columns feature_field output_field))]
         (log/info {:msg "Bucketizer completed successfully"})
         {:data preview})
       (catch Exception err
         (throw (ex-info "Bucketizer execution failed"
                         {:type :algorithm/bucketizer-failed
                          :feature-field feature_field
                          :error (.getMessage err)}
                         err)))))