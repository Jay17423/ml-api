(ns ml-api.services.dataset
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.sql.types StructField
    StructType
    Metadata
    DataTypes]))


(defn create-schema
  "Creates dynamic schema using feature fields."
  [feature-cols]
  (StructType. (into-array StructField (map (fn [col-name]
                                              (StructField.
                                               col-name
                                               DataTypes/DoubleType
                                               true
                                               (Metadata/empty)))
                                            feature-cols))))

(defn read-dataset
  [spark {:keys [url options feature-field]}]

  (let [opts (merge {:header true :delimiter ","} options)
        schema (create-schema feature-field)]
    (try
      (log/info
       {:msg "Reading dataset"
        :metric {:path url}})

      (-> spark
          .read
          (.format "csv")
          (.schema schema)
          (.option "header"
                   (if (:header opts) "true" "false"))
          (.option "delimiter"
                   (:delimiter opts))
          (.load url))

      (catch Exception err
        (throw (ex-info "Failed to read dataset"
                        {:path url
                         :error (.getMessage err)}
                        err))))))