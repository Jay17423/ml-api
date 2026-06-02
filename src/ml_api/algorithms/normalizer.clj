(ns ml-api.algorithms.normalizer
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va])
  (:import
   [org.apache.spark.ml.feature Normalizer]
   [org.apache.spark.ml.linalg Vector]))

(defn parse-norm-value
  [norm-value]
  (if (= norm-value "inf") Double/POSITIVE_INFINITY
      (double norm-value)))

(defn vector->clojure
  [vector]
  (vec (.toArray ^Vector vector)))

(defn row->clojure
  [row output-field]
  {:normalized-features (vector->clojure (.getAs row output-field))})

(defn dataset->json
  [dataset output-field]
  (mapv #(row->clojure % output-field) (.collectAsList dataset)))

(defn execute
  "Executes Spark Normalizer."
  [dataset {:keys [feature_field
                   output_field
                   norm_value]
            :or {output_field "normalized_features"
                 norm_value 2.0}}]

  (try
    (log/info {:msg "Starting Normalizer"
               :feature-field feature_field
               :output-field output_field
               :norm-value norm_value})

    (let [vectorized-dataset (va/create-feature-vector dataset feature_field)
          normalizer (-> (Normalizer.)
                         (.setInputCol "features")
                         (.setOutputCol output_field)
                         (.setP (parse-norm-value norm_value)))
          transformed-dataset (.transform normalizer vectorized-dataset)
          preview (dataset->json transformed-dataset output_field)]
      (log/info {:msg "Normalizer completed successfully"})
      {:norm-type norm_value
       :data preview})
    (catch Exception err
      (throw (ex-info "Normalizer execution failed"
                      {:type :algorithm/normalizer-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))