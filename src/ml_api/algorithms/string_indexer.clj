(ns ml-api.algorithms.string-indexer
  "Spark StringIndexer implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature StringIndexer]))

(defn transform
  "Transforms categorical values into indices."
  [ds input-fields output-fields invalid-value label-order]

  (let [indexer (-> (StringIndexer.)
                    (.setInputCols (into-array String input-fields))
                    (.setOutputCols (into-array String output-fields))
                    (.setHandleInvalid invalid-value)
                    (.setStringOrderType label-order))
        model (.fit indexer ds)]
    (.transform model ds)))

(defn execute
  "Executes Spark StringIndexer."
  [ds {:keys [input_fields output_fields invalid_value label_order]
       :or {invalid_value "error" label_order "frequencyDesc"}}]

  (try
    (log/info {:msg "Starting StringIndexer"
               :input-fields input_fields
               :output-fields output_fields
               :invalid-value invalid_value
               :label-order label_order})

    (let [transformed-ds (transform ds input_fields output_fields invalid_value
                                    label_order)
          preview (utils/dataset->json
                   transformed-ds
                   (utils/preview-columns input_fields output_fields))]
      (log/info {:msg "StringIndexer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "StringIndexer execution failed"
                      {:type :algorithm/string-indexer-failed
                       :input-fields input_fields
                       :error (.getMessage err)}
                      err)))))