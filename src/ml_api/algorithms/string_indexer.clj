(ns ml-api.algorithms.string-indexer
  "Spark StringIndexer implementation."
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature StringIndexer]))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row input-fields output-fields]
  (merge (into {} (map (fn [field] [field (.getAs row field)]) input-fields))
         (into {} (map (fn [field] [field (.getAs row field)]) output-fields))))

(defn dataset->json
  "Converts dataset into JSON preview."
  [dataset input-fields output-fields]
  (mapv #(row->clojure % input-fields output-fields) (.collectAsList dataset)))

(defn transform
  "Transforms categorical values into indices."
  [dataset input-fields output-fields invalid-value label-order]
  (let [indexer (-> (StringIndexer.)
                    (.setInputCols (into-array String input-fields))
                    (.setOutputCols (into-array String output-fields))
                    (.setHandleInvalid invalid-value)
                    (.setStringOrderType label-order))
        model (.fit indexer dataset)]
    (.transform model dataset)))


(defn execute
  "Executes Spark StringIndexer."
  [dataset {:keys [input_fields output_fields invalid_value label_order]
            :or {invalid_value "error" label_order "frequencyDesc"}}]
  (try
    (log/info {:msg "Starting StringIndexer"
               :input-fields input_fields
               :output-fields output_fields
               :invalid-value invalid_value
               :label-order label_order})

    (let [trans-dataset (transform dataset
                                   input_fields
                                   output_fields
                                   invalid_value
                                   label_order)
          preview (dataset->json trans-dataset input_fields output_fields)]

      (log/info {:msg "StringIndexer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "StringIndexer execution failed"
                      {:type :algorithm/string-indexer-failed
                       :input-fields input_fields
                       :error (.getMessage err)}
                      err)))))