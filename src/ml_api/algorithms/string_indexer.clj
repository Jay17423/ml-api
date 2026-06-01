(ns ml-api.algorithms.string-indexer
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature StringIndexer]))

(defn row->clojure
  [row input-fields output-fields]
  (merge (into {} (map (fn [field] [field (.getAs row field)]) input-fields))
         (into {} (map (fn [field] [field (.getAs row field)]) output-fields))))

(defn dataset->json
  [dataset input-fields output-fields]
  (mapv #(row->clojure % input-fields output-fields) (.collectAsList dataset)))

(defn execute
  "Executes Spark StringIndexer."
  [dataset {:keys [input_fields
                   output_fields
                   invalid_value
                   label_order]
            :or {invalid_value "error"
                 label_order "frequencyDesc"}}]

  (try
    (log/info {:msg "Starting StringIndexer"
               :input-fields input_fields
               :output-fields output_fields
               :invalid-value invalid_value
               :label-order label_order})

    (let [indexer (-> (StringIndexer.)
                      (.setInputCols (into-array String input_fields))
                      (.setOutputCols (into-array String output_fields))
                      (.setHandleInvalid invalid_value)
                      (.setStringOrderType label_order))
          model (.fit indexer dataset)
          transformed-dataset (.transform model dataset)
          preview (dataset->json transformed-dataset input_fields output_fields)]
      (log/info {:msg "StringIndexer completed successfully"})
      {:labels (mapv vec (.labelsArray model))
       :data preview})
    (catch Exception err
      (throw (ex-info "StringIndexer execution failed"
                      {:type :algorithm/string-indexer-failed
                       :input-fields input_fields
                       :error (.getMessage err)}
                      err)))))