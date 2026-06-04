(ns ml-api.algorithms.standard-scaler
  "Spark StandardScaler implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature StandardScaler]))

(defn row->clojure
  "Converts Spark row into Clojure map."
  [row output-field]
  {:scaled-features (utils/vector->clojure (.getAs row output-field))})

(defn dataset->json
  "Converts dataset into JSON preview."
  [ds output-field]
  (mapv #(row->clojure % output-field) (.collectAsList ds)))

(defn transform
  "Scales feature vectors."
  [ds feature-field output-field with-std with-mean]
  (let [vectorized-ds (va/create-feature-vector ds feature-field)
        scaler (-> (StandardScaler.)
                   (.setInputCol "features")
                   (.setOutputCol output-field)
                   (.setWithStd with-std)
                   (.setWithMean with-mean))
        model (.fit scaler vectorized-ds)]
    (.transform model vectorized-ds)))

(defn execute
  "Executes Spark StandardScaler."
  [ds {:keys [feature_field output_field with_std with_mean]
       :or {output_field "scaled_features" with_std true with_mean false}}]
  (try
    (log/info {:msg "Starting StandardScaler"
               :feature-field feature_field
               :output-field output_field
               :with-std with_std
               :with-mean with_mean})
    (let [transformed-ds (transform ds feature_field output_field with_std
                                    with_mean)
          preview (dataset->json transformed-ds output_field)]
      (log/info {:msg "StandardScaler completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "StandardScaler execution failed"
                      {:type :algorithm/standard-scaler-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))