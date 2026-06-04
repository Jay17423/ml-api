(ns ml-api.algorithms.chi-square-test
  "Spark Chi-Square_Test Implementation"
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.stat ChiSquareTest]))

(defn parse-result
  "Parses Spark ChiSquareTest result."
  [result-row flatten]
  (let [p-values (.getAs result-row "pValues")
        degrees-of-freedom (.getAs result-row "degreesOfFreedom")
        statistics (.getAs result-row "statistics")]

    (if flatten
      {:p-values (utils/vector->clojure p-values)
       :degrees-of-freedom (utils/wrapped-array->clojure degrees-of-freedom)
       :statistics (utils/vector->clojure statistics)}
      {:p-values p-values
       :degrees-of-freedom degrees-of-freedom
       :statistics statistics})))

(defn execute
  "Executes Spark ChiSquareTest."
  [ds {:keys [label_field feature_field flatten] :or {flatten true}}]
  (try
    (let [vectorized-ds (va/create-feature-vector ds feature_field)
          result-df (ChiSquareTest/test vectorized-ds "features" label_field)
          result-row (.head result-df)]
      (log/info {:msg "ChiSquareTest completed successfully"
                 :label-field label_field
                 :flatten flatten})
      {:data (parse-result result-row flatten)})
    (catch Exception err
      (throw (ex-info "ChiSquareTest execution failed"
                      {:type :algorithm/chi-square-test-failed
                       :label-field label_field
                       :error (.getMessage err)}
                      err)))))