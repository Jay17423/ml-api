(ns ml-api.algorithms.normalizer
  "Spark Normalizer implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature Normalizer]))

(defn parse-norm-value
  "Convert inf into numerical infinity otherwise normal numerical value "
  [norm-value]
  (if (= norm-value "inf")
    Double/POSITIVE_INFINITY
    (double norm-value)))

(defn transform
  "Normalizes feature vectors."
  [ds feature-field output-field norm-value]
  (let [vectorized-ds (va/create-feature-vector ds feature-field)
        normalizer (-> (Normalizer.)
                       (.setInputCol "features")
                       (.setOutputCol output-field)
                       (.setP (parse-norm-value norm-value)))]
    (.transform normalizer vectorized-ds)))

(defn execute
  "Executes Spark Normalizer."
  [ds {:keys [feature_field output_field norm_value]
       :or {output_field "normalized_features" norm_value 2.0}}]
  (try
    (log/info {:msg "Starting Normalizer"
               :feature-field feature_field
               :output-field output_field
               :norm-value norm_value})
    (let [transformed-ds (transform ds feature_field output_field norm_value)
          preview (utils/dataset->json transformed-ds [output_field])]
      (log/info {:msg "Normalizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "Normalizer execution failed"
                      {:type :algorithm/normalizer-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))