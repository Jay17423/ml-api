(ns ml-api.algorithms.binarizer
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature Binarizer]))

(defn row->clojure
  [row input-fields output-fields]
  (merge (into {} (map (fn [field] [field (.getAs row field)]) input-fields)) 
         (into {} (map (fn [field] [field (.getAs row field)]) output-fields))))

(defn dataset->json
  [dataset input-fields output-fields]
  (mapv #(row->clojure % input-fields output-fields) (.collectAsList dataset)))

(defn execute
  "Executes Spark Binarizer."
  [dataset {:keys [input_field
                   output_field
                   threshold_values]
            :or {threshold_values [0.0]}}]

  (try
    (log/info {:msg "Starting Binarizer"
               :input-field input_field
               :output-field output_field
               :threshold-values threshold_values}) 
    (let [binarizer (-> (Binarizer.)
                        (.setInputCols (into-array String input_field))
                        (.setOutputCols (into-array String output_field))
                        (.setThresholds (double-array threshold_values)))
          transformed-dataset (.transform binarizer dataset)
          preview (dataset->json transformed-dataset input_field output_field)]
      (log/info {:msg "Binarizer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "Binarizer execution failed"
                      {:type :algorithm/binarizer-failed
                       :input-field input_field
                       :error (.getMessage err)}
                      err)))))