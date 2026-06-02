(ns ml-api.algorithms.imputer
  (:require
   [taoensso.timbre :as log]
   [flambo.sql-functions :as sqlf])
  (:import
   [org.apache.spark.ml.feature Imputer]
   [org.apache.spark.sql Dataset Column]))

(defn replace-custom-missing-values
  [dataset feature-fields missing-values]
  (reduce (fn [df column]
            (reduce (fn [updated-df missing-val]
                      (.withColumn
                       ^Dataset updated-df column
                       (.cast (.otherwise (org.apache.spark.sql.functions/when
                                           (.equalTo
                                            ^Column (sqlf/col column)
                                            missing-val)
                                            nil)
                                          (sqlf/col column))
                              "double")))
                    df
                    missing-values))
          dataset
          feature-fields))

(defn row->clojure
  [row feature-fields output-fields]

  (merge (into {} (map (fn [field] [field (.getAs row field)])
                       feature-fields)) 
         (into {} (map (fn [field] [field (.getAs row field)])
                       output-fields))))

(defn dataset->json
  [dataset feature-fields output-fields]

  (mapv #(row->clojure % feature-fields output-fields)
        (.collectAsList dataset)))

(defn execute
  "Executes Spark Imputer."
  [dataset {:keys [feature_field
                   output_field
                   strategy
                   missing_value
                   relative_error]

            :or {strategy "mean"
                 missing_value ["NaN"]
                 relative_error 0.0001}}]

  (try
    (log/info {:msg "Starting Imputer"
               :feature-field feature_field
               :output-field output_field
               :strategy strategy
               :missing-value missing_value
               :relative-error relative_error})
    (let [processed-dataset (replace-custom-missing-values
                             dataset
                             feature_field
                             missing_value)
          imputer (-> (Imputer.)
                      (.setInputCols (into-array String feature_field))
                      (.setOutputCols (into-array String output_field))
                      (.setStrategy strategy)
                      (.setRelativeError (double relative_error)))

          model (.fit imputer processed-dataset)

          transformed-dataset
          (.transform model
                      processed-dataset)

          preview
          (dataset->json
           transformed-dataset
           feature_field
           output_field)]
      (log/info {:msg "Imputer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "Imputer execution failed"
                      {:type :algorithm/imputer-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))