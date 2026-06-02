(ns ml-api.algorithms.standard-scaler
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va])
  (:import
   [org.apache.spark.ml.feature StandardScaler]
   [org.apache.spark.ml.linalg Vector]))

(defn vector->clojure
  [vector]
  (vec (.toArray ^Vector vector)))

(defn row->clojure
  [row output-field]
  {:scaled-features (vector->clojure (.getAs row output-field))})

(defn dataset->json
  [dataset output-field]
  (mapv #(row->clojure % output-field) (.collectAsList dataset)))

(defn execute
  "Executes Spark StandardScaler."

  [dataset {:keys [feature_field
                   output_field
                   with_std
                   with_mean]
            :or {output_field "scaled_features"
                 with_std true
                 with_mean false}}]

  (try
    (log/info {:msg "Starting StandardScaler"
               :feature-field feature_field
               :output-field output_field
               :with-std with_std
               :with-mean with_mean})
    (let [vectorized-dataset (va/create-feature-vector dataset feature_field)
          scaler (-> (StandardScaler.)
                     (.setInputCol "features")
                     (.setOutputCol output_field)
                     (.setWithStd with_std)
                     (.setWithMean with_mean))
          model (.fit scaler vectorized-dataset)
          transformed-dataset (.transform model vectorized-dataset)
          preview (dataset->json transformed-dataset output_field)]
      (log/info {:msg "StandardScaler completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "StandardScaler execution failed"
                      {:type :algorithm/standard-scaler-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))