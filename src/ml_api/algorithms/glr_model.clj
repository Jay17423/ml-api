(ns ml-api.algorithms.glr-model
  "Spark GeneralizedLinearRegressionModel implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va])
  (:import
   [org.apache.spark.ml.regression GeneralizedLinearRegressionModel]))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row target-field output-field]
  {(keyword target-field) (.getAs row target-field)
   (keyword output-field) (.getAs row output-field)})

(defn dataset->json
  "Converts dataset into JSON preview."
  [ds target-field output-field]
  (mapv #(row->clojure % target-field output-field) (.collectAsList ds)))

(defn transform
  "Loads trained model and generates predictions."
  [ds model-path feature-field]
  (let [vectorized-ds (va/create-feature-vector ds feature-field)
        model (GeneralizedLinearRegressionModel/load model-path)]
    (.transform model vectorized-ds)))

(defn execute
  "Executes Spark GeneralizedLinearRegressionModel."
  [ds {:keys [model_path feature_field target_field output_field]
       :or {target_field "label" output_field "prediction"}}]

  (try
    (log/info {:msg "Starting GeneralizedLinearRegressionModel"
               :model-path model_path
               :feature-field feature_field
               :target-field target_field
               :output-field output_field})

    (let [transformed-ds (transform ds model_path feature_field)
          preview (dataset->json transformed-ds target_field output_field)]
      (log/info
       {:msg "GeneralizedLinearRegressionModel completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info
              "GeneralizedLinearRegressionModel execution failed"
              {:type :algorithm/generalized-linear-regression-model-failed
               :feature-field feature_field
               :model-path model_path
               :error (.getMessage err)}
              err)))))