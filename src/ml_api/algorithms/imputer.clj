(ns ml-api.algorithms.imputer
  "Spark Imputer implementation."
  (:require
   [taoensso.timbre :as log]
   [flambo.sql-functions :as sqlf])
  (:import
   [org.apache.spark.ml.feature Imputer]
   [org.apache.spark.sql Dataset Column]))

(defn replace-values
  "Replaces custom missing values with null."
  [ds feature-fields missing-values]
  (reduce (fn [df column]
            (reduce (fn [updated-df missing-val]
                      (.withColumn ^Dataset updated-df column
                                   (.cast (.otherwise
                                           (org.apache.spark.sql.functions/when
                                            (.equalTo ^Column
                                             (sqlf/col column)
                                                      missing-val)
                                             nil)
                                           (sqlf/col column))
                                          "double")))
                    df
                    missing-values))
          ds
          feature-fields))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row feature-fields output-fields]
  (merge (into {} (map (fn [field] [field (.getAs row field)]) feature-fields))
         (into {} (map (fn [field] [field (.getAs row field)]) output-fields))))

(defn dataset->json
  "Converts dataset into JSON preview."
  [ds feature-fields output-fields]
  (mapv #(row->clojure % feature-fields output-fields) (.collectAsList ds)))

(defn transform
  "Imputes missing values."
  [ds feature-fields output-fields strategy missing-values relative-error]
  (let [processed-ds (replace-values ds feature-fields missing-values)
        imputer (-> (Imputer.)
                    (.setInputCols (into-array String feature-fields))
                    (.setOutputCols (into-array String output-fields))
                    (.setStrategy strategy)
                    (.setRelativeError (double relative-error)))
        model (.fit imputer processed-ds)]
    (.transform model processed-ds)))

(defn execute
  "Executes Spark Imputer."
  [ds {:keys [feature_field output_field strategy missing_value relative_error]
       :or {strategy "mean" missing_value ["NaN"] relative_error 0.0001}}]
  (try
    (log/info {:msg "Starting Imputer"
               :feature-field feature_field
               :output-field output_field
               :strategy strategy
               :missing-value missing_value
               :relative-error relative_error})
    (let [transformed-ds (transform ds feature_field output_field
                                    strategy
                                    missing_value
                                    relative_error)
          preview (dataset->json transformed-ds feature_field output_field)]
      (log/info {:msg "Imputer completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "Imputer execution failed"
                      {:type :algorithm/imputer-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))