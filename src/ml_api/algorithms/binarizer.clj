(ns ml-api.algorithms.binarizer
  "Spark Binarizer implementation."
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature Binarizer]))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row input-fields output-fields]
  (merge (into {} (map (fn [field] [field (.getAs row field)]) input-fields))
         (into {} (map (fn [field] [field (.getAs row field)]) output-fields))))

(defn dataset->json
  "Converts dataset into JSON preview."
  [ds input-fields output-fields]
  (mapv #(row->clojure % input-fields output-fields) (.collectAsList ds)))

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
          preview (dataset->json transformed-ds input_field output_field)]
      (log/info {:msg "Binarizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "Binarizer execution failed"
                      {:type :algorithm/binarizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))