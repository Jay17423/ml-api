(ns ml-api.middleware
  "Middleware funtion for the Ml-Api Application."
  (:require
   [ring.util.response :refer [response status]]
   [taoensso.timbre :as log]))

(defn error-response
  "Returns the error response to the API request"
  [http-status message duration details]
  (-> (response {:status "error"
                 :message message
                 :details details
                 :duration-ms duration})
      (status http-status)))

(defn status-for-error
  "Returns correct HTTP status code according to the error."
  [error-type]
  (case error-type
    :validation/invalid-algorithm 404
    :validation/unsupported-algorithm 404
    :algorithm/not-supported 404
    :dataset/read-failed 404
    :validation/invalid-parameters 400
    :validation/invalid-parameter-type 400
    :validation/missing-parameter 400
    :validation/invalid-column 400
    :preprocessing/dataset-failed 400
    500))

(defn wrap-error-handler
  "Centralized exception handling middleware."
  [handler]
  (fn [req]
    (try
      (handler req)
      (catch AssertionError err
        (let [duration (- (System/currentTimeMillis)
                          (:start-time req))]
          (log/warn {:msg "Validation failed"
                     :error (.getMessage err)
                     :metric {:duration-ms duration}})
          (error-response 400
                          "Validation failed"
                          duration
                          (.getMessage err))))
      (catch clojure.lang.ExceptionInfo err
        (let [duration (- (System/currentTimeMillis) (:start-time req))
              err-data (ex-data err)
              error-type (:type err-data)
              http-status (status-for-error error-type)
              message
              (case error-type
                :validation/invalid-algorithm
                "Invalid algorithm"

                :validation/unsupported-algorithm
                "Unsupported algorithm"

                :validation/invalid-parameters
                "Invalid parameters"

                :validation/invalid-parameter-type
                "Invalid parameter type"

                :validation/missing-parameter
                "Missing required parameter"

                :validation/invalid-column
                "Invalid column"

                :dataset/read-failed
                "Unable to read dataset"

                :preprocessing/dataset-failed
                "Dataset preprocessing failed"

                :vector/vector-creation-failed
                "Feature vector creation failed"

                :spark/session-create-failed
                "Spark session creation failed"

                :spark/warmup-failed
                "Spark warmup failed"

                "Application error")]
          (log/error {:msg message
                      :type error-type
                      :details err-data
                      :error (.getMessage err)
                      :metric {:duration-ms duration}})
          (error-response http-status
                          message
                          duration
                          err-data)))

      (catch Exception err
        (let [duration (- (System/currentTimeMillis) (:start-time req))]
          (log/error {:msg "Unhandled exception"
                      :error (.getMessage err)
                      :exception err
                      :metric {:duration-ms duration}})
          (error-response 500
                          "Internal server error"
                          duration
                          (.getMessage err)))))))

(defn wrap-request-timer
  "Attaches start-time to request for duration tracking."
  [handler]
  (fn [req]
    (handler (assoc req :start-time (System/currentTimeMillis)))))