(ns ml-api.services.dataset
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.sql.types StructField StructType Metadata DataTypes]))


(defn create-schema
  [all-cols]
  (StructType. (into-array StructField
                           (map (fn [col-name]
                                  (StructField.
                                   col-name
                                   DataTypes/DoubleType
                                   true
                                   (Metadata/empty)))
                                all-cols))))

(defn read-dataset
  [spark {:keys [url options feature_field label_field]}]

  (let [opts (merge {:header true :delimiter ","} options)
        all-cols (conj (vec feature_field) label_field)
        schema (create-schema all-cols)]
    (try
      (log/info {:msg "Reading dataset"
                 :url url
                 :columns all-cols})
      (-> spark
          .read
          (.format "csv")
          (.schema schema)
          (.option "header" (if (:header opts) "true" "false"))
          (.option "delimiter" (or (:delimiter opts) ","))
          (.load url))
      (catch Exception err
        (throw (ex-info "Dataset read failed"
                        {:type :dataset/read-failed
                         :url url
                         :columns all-cols
                         :error (.getMessage err)}
                        err))))))