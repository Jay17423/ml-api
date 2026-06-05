(ns ml-api.algorithms.binarizer
  "Spark Binarizer implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature Binarizer]))

(defn transform
  "Transforms numeric values into binary values."
  [ds input-fields output-fields threshold-values]
  (let [binarizer (-> (Binarizer.)
                      (.setInputCols (into-array String input-fields))
                      (.setOutputCols (into-array String output-fields))
                      (.setThresholds (double-array threshold-values)))]
    (.transform binarizer ds)))

(defn execute
  "Executes Spark Binarizer."
  [ds {:keys [input_field output_field threshold_values]
       :or {threshold_values [0.0]}}]
  (try
    (log/info {:msg "Starting Binarizer"
               :input-field input_field
               :output-field output_field
               :threshold-values threshold_values})
    (let [transformed-ds (transform ds input_field output_field
                                    threshold_values)
          preview (utils/dataset->json transformed-ds
                                       (utils/preview-columns
                                        input_field
                                        output_field))]
      (log/info {:msg "Binarizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "Binarizer execution failed"
                      {:type :algorithm/binarizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))