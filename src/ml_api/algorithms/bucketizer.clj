(ns ml-api.algorithms.bucketizer
  "Spark Bucketizer implementation."
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature Bucketizer]))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row feature-fields output-fields]
  (merge (into {} (map (fn [field] [field (.getAs row field)]) feature-fields))
         (into {} (map (fn [field] [field (.getAs row field)]) output-fields))))

(defn dataset->json
  "Converts dataset into JSON preview."
  [dataset feature-fields output-fields]
  (mapv #(row->clojure % feature-fields output-fields) (.collectAsList dataset)))

(defn splits->2d-array
  "Converts bucket splits into 2D double array."
  [bucket-splits]
  (into-array (map double-array bucket-splits)))

(defn transform
  "Transforms numeric values into buckets."
  [dataset feature-fields output-fields bucket-splits invalid-value]
  (let [bucketizer (-> (Bucketizer.)
                       (.setInputCols (into-array String feature-fields))
                       (.setOutputCols (into-array String output-fields))
                       (.setSplitsArray (splits->2d-array bucket-splits))
                       (.setHandleInvalid invalid-value))]
    (.transform bucketizer dataset)))

(defn execute
  "Executes Spark Bucketizer."
  [dataset {:keys [feature_field output_field bucket_splits invalid_value]
            :or {invalid_value "error"}}]

  (try
    (log/info {:msg "Starting Bucketizer"
               :feature-field feature_field
               :output-field output_field
               :bucket-splits bucket_splits
               :invalid-value invalid_value})

    (let [transformed-dataset (transform dataset feature_field output_field
                                         bucket_splits
                                         invalid_value)

          preview (dataset->json transformed-dataset feature_field
                                 output_field)]
      (log/info {:msg "Bucketizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "Bucketizer execution failed"
                      {:type :algorithm/bucketizer-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))