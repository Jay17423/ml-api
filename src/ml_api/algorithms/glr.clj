(ns ml-api.algorithms.glr
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va])
  (:import
   [org.apache.spark.ml.regression GeneralizedLinearRegression]))


(defn row->clojure
  [row target-field output-field]

  {(keyword target-field) (.getAs row target-field)
   (keyword output-field) (.getAs row output-field)})

(defn dataset->json
  [ds target-field output-field]
  (mapv #(row->clojure % target-field output-field) (.collectAsList ds)))

(defn execute
  "Executes Spark GeneralizedLinearRegression."

  [ds {:keys [feature_field
              target_field
              output_field
              model_path
              regression_family
              prediction_link
              include_intercept
              max_iterations
              training_tolerance
              regularization_strength
              row_weight_field
              training_solver
              link_output_field
              offset_field
              aggregation_depth]
       :or {output_field "prediction"
            regression_family "gaussian"
            include_intercept true
            max_iterations 25
            training_tolerance 0.000001
            regularization_strength 0.0
            training_solver "irls"
            aggregation_depth 2}}]

  (try
    (log/info {:msg "Starting GeneralizedLinearRegression"
               :feature-field feature_field
               :target-field target_field
               :output-field output_field
               :family regression_family})
    (let [vectorized-ds (va/create-feature-vector ds feature_field)
          glr (-> (GeneralizedLinearRegression.)
                  (.setFeaturesCol "features")
                  (.setLabelCol target_field)
                  (.setPredictionCol output_field)
                  (.setFamily regression_family)
                  (.setFitIntercept include_intercept)
                  (.setMaxIter max_iterations)
                  (.setTol (double training_tolerance))
                  (.setRegParam (double regularization_strength))
                  (.setSolver training_solver)
                  (.setAggregationDepth aggregation_depth))
          glr (if prediction_link (.setLink glr prediction_link)
                  glr)

          glr (if row_weight_field (.setWeightCol glr row_weight_field)
                  glr)

          glr (if link_output_field (.setLinkPredictionCol glr link_output_field)
                  glr)

          glr (if offset_field (.setOffsetCol glr offset_field)
                  glr)

          model (.fit glr vectorized-ds)
          _ (when model_path
              (.save (.write model) model_path))
          transformed-ds (.transform model vectorized-ds)
          preview (dataset->json transformed-ds target_field output_field)]
      (log/info {:msg "GeneralizedLinearRegression completed successfully"})
      {:data preview})
    (catch Exception err
      (throw (ex-info "GeneralizedLinearRegression execution failed"
                      {:type :algorithm/generalized-linear-regression-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))