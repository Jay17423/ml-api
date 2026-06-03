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
  [dataset target-field output-field]
  (mapv #(row->clojure % target-field output-field) (.collectAsList dataset)))

(defn transform
  "Loads trained model and generates predictions."
  [dataset model-path feature-field]
  (let [vectorized-dataset (va/create-feature-vector dataset feature-field)
        model (GeneralizedLinearRegressionModel/load model-path)]
    (.transform model vectorized-dataset)))

(defn execute
  "Executes Spark GeneralizedLinearRegressionModel."
  [dataset {:keys [model_path feature_field target_field output_field]
            :or {target_field "label" output_field "prediction"}}]

  (try
    (log/info {:msg "Starting GeneralizedLinearRegressionModel"
               :model-path model_path
               :feature-field feature_field
               :target-field target_field
               :output-field output_field})

    (let [transformed-dataset (transform dataset model_path feature_field)
          preview (dataset->json transformed-dataset target_field output_field)]
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