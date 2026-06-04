(ns ml-api.preprocessing
  "Dataset preprocessing utilities."
  (:require
   [clojure.set :as set]
   [taoensso.timbre :as log]
   [flambo.sql-functions :as sqlf])
  (:import
   [org.apache.spark.sql Dataset]
   [org.apache.spark.sql.types IntegerType LongType FloatType StructField]))

(defn normalize-columns
  "Normalizes string/vector columns into vector."
  [value]
  (cond
    (string? value)
    [value]

    (vector? value)
    value

    :else
    []))

(defn get-required-columns
  "Extracts required columns from parameters."
  [params]
  (vec (distinct (concat (normalize-columns
                          (:feature_field params))

                         (normalize-columns
                          (:input_field params))

                         (normalize-columns
                          (:text_field params))

                         (normalize-columns
                          (:target_field params))))))

(defn validate-columns
  "Validates whether required columns exist."
  [ds required-columns]
  (let [dataset-columns (set (.fieldNames (.schema ^Dataset ds)))
        missing-columns (set/difference (set required-columns) dataset-columns)]
    (when (seq missing-columns)
      (throw
       (ex-info "Column validation failed"
                {:type :validation/invalid-column
                 :missing-columns missing-columns})))
    ds))

(defn cast-column-to-double
  "Casts numeric column into double."
  [ds field]
  (.withColumn ^Dataset ds field (.cast (sqlf/col field) "double")))

(defn preprocess-dataset
  "Validates and preprocesses dataset."
  [ds parameters]
  (try
    (let [required-columns (get-required-columns parameters)
          _ (validate-columns ds required-columns)
          schema-fields (.fields (.schema ^Dataset ds))
          processed-dataset
          (reduce (fn [df ^StructField field]
                    (if (and (contains? (set required-columns) (.name field))
                             (or (instance? IntegerType (.dataType field))
                                 (instance? LongType (.dataType field))
                                 (instance? FloatType (.dataType field))))
                      (cast-column-to-double df (.name field))
                      df))
                  ds
                  schema-fields)]
      (log/info {:msg "Dataset preprocessing completed successfully"
                 :required-columns required-columns})
      processed-dataset)
    (catch Exception err
      (throw (ex-info "Dataset preprocessing failed"
                      {:type :preprocessing/dataset-failed
                       :error (.getMessage err)}
                      err)))))