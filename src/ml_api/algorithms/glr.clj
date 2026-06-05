(ns ml-api.algorithms.glr
  "Spark GeneralizedLinearRegression implementation."
  (:require
   [clj-time.core :as time]
   [clj-time.coerce :as tc]
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils])
  (:import
   [java.util UUID]
   [org.apache.spark.ml.regression GeneralizedLinearRegression]))

(defn dataset-preview
  "Converts Spark dataset into JSON response preview."
  [ds target-field output-field]
  (utils/dataset->json ds [target-field output-field]))

(defn current-time-ms
  "Returns current timestamp in milliseconds."
  []
  (tc/to-long (time/now)))

(defn calculate-duration-sec
  "Calculates duration in seconds."
  [start-time end-time]
  (float (/ (- end-time start-time) 1000.0)))

(defn validate-train-size
  "Validates train size percentage."
  [train-size]
  (when (or (<= train-size 0) (>= train-size 100))
    (throw
     (ex-info "Invalid train size"
              {:type :validation/invalid-parameter
               :parameter :train_size
               :value train-size}))))

(defn create-glr
  "Creates configured GeneralizedLinearRegression instance."
  [{:keys [target_field
           output_field
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
           aggregation_depth]}]

  (let [glr (-> (GeneralizedLinearRegression.)
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

        glr (if prediction_link
              (.setLink glr prediction_link)
              glr)

        glr (if row_weight_field
              (.setWeightCol glr row_weight_field)
              glr)

        glr (if link_output_field
              (.setLinkPredictionCol glr link_output_field)
              glr)]

    (if offset_field
      (.setOffsetCol glr offset_field)
      glr)))

(defn save-model
  "Saves trained model if model path is provided."
  [model model-path]
  (when model-path
    (.save
     (.overwrite (.write model))
     model-path)))

(defn execute
  "Executes Spark GeneralizedLinearRegression."
  [ds {:keys [feature_field
              target_field
              output_field
              model_path
              model_label
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
              aggregation_depth
              train_size
              seed]
       :or {output_field "prediction"
            regression_family "gaussian"
            include_intercept true
            max_iterations 25
            training_tolerance 0.000001
            regularization_strength 0.0
            training_solver "irls"
            aggregation_depth 2
            train_size 80
            seed 123}}]

  (try
    (validate-train-size train_size)
    (log/info {:msg "Starting GeneralizedLinearRegression"
               :feature-field feature_field
               :target-field target_field
               :output-field output_field
               :family regression_family
               :train-size train_size
               :seed seed})

    (let [vectorized-ds (va/create-feature-vector ds feature_field)
          train-ratio (/ train_size 100.0)
          test-ratio (- 1.0 train-ratio)
          split-datasets (.randomSplit vectorized-ds
                                       (double-array [train-ratio test-ratio])
                                       (long seed))
          train-ds (aget split-datasets 0)
          test-ds (aget split-datasets 1)
          train-start (current-time-ms)
          glr (create-glr {:target_field target_field
                           :output_field output_field
                           :regression_family regression_family
                           :prediction_link prediction_link
                           :include_intercept include_intercept
                           :max_iterations max_iterations
                           :training_tolerance training_tolerance
                           :regularization_strength regularization_strength
                           :row_weight_field row_weight_field
                           :training_solver training_solver
                           :link_output_field link_output_field
                           :offset_field offset_field
                           :aggregation_depth aggregation_depth})
          model (.fit glr train-ds)
          train-end (current-time-ms)
          _ (save-model model model_path)
          prediction-start (current-time-ms)
          transformed-ds (.transform model test-ds)
          prediction-end (current-time-ms)
          preview (dataset-preview transformed-ds target_field output_field)]
      (log/info
       {:msg "GeneralizedLinearRegression completed successfully"})
      {:model-id (str (UUID/randomUUID))
       :model-label model_label
       :model-algo (.getSimpleName (.getClass model))
       :regression-family regression_family
       :train-size train_size
       :seed seed
       :coefficients (vec (.toArray (.coefficients model)))
       :intercept (.intercept model)
       :training-time-sec (calculate-duration-sec train-start train-end)
       :prediction-time-sec (calculate-duration-sec prediction-start
                                                    prediction-end)
       :data preview})
    (catch Exception err
      (throw (ex-info "GeneralizedLinearRegression execution failed"
                      {:type :algorithm/generalized-linear-regression-failed
                       :feature-field feature_field
                       :error (.getMessage err)}
                      err)))))