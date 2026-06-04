(ns ml-api.algorithms.chi-sq-selector
  "Spark Chi-Sq-Selector Implementations"
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.feature ChiSqSelector]))

(defn row->clojure
  "Convert Spark row into clojure map"
  [row output-field]
  {:selected-features (utils/vector->clojure (.getAs row output-field))})

(defn dataset->json
  "Convert Dataset into JSON preview"
  [ds output-field]
  (mapv #(row->clojure % output-field) (.collectAsList ds)))

(defn execute
  "Executes Spark ChiSqSelector."
  [ds {:keys [feature_field target_field output_field selection_method
              top_feature
              selection_percentage
              fpr_threshold
              fdr_threshold
              fwe_threshold]
       :or {target_field "label" selection_method "numTopFeatures"
            top_feature 50
            selection_percentage 0.1
            fpr_threshold 0.05
            fdr_threshold 0.05
            fwe_threshold 0.05
            output_field "selected_features"}}]

  (try
    (log/info {:msg "Starting ChiSqSelector"
               :feature-field feature_field
               :target-field target_field
               :output-field output_field
               :selection-method selection_method})
    (let [vectorized-ds (va/create-feature-vector ds feature_field)
          selector (-> (ChiSqSelector.)
                       (.setFeaturesCol "features")
                       (.setLabelCol target_field)
                       (.setOutputCol output_field)
                       (.setSelectorType selection_method))
          selector (case selection_method
                     "numTopFeatures"
                     (.setNumTopFeatures
                      selector
                      top_feature)

                     "percentile"
                     (.setPercentile
                      selector
                      (double selection_percentage))

                     "fpr"
                     (.setFpr
                      selector
                      (double fpr_threshold))

                     "fdr"
                     (.setFdr
                      selector
                      (double fdr_threshold))

                     "fwe"
                     (.setFwe
                      selector
                      (double fwe_threshold))

                     selector)
          model (.fit selector vectorized-ds)
          transformed-ds (.transform model vectorized-ds)
          preview (dataset->json transformed-ds output_field)]
      (log/info {:msg "ChiSqSelector completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "ChiSqSelector execution failed"
                      {:type :algorithm/chisq-selector-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))