(ns ml-api.middleware
  (:require
   [ring.util.response :refer [response status]]
   [taoensso.timbre :as log]))

(defn error-response
  [http-status message duration details]
  (-> (response
       {:status "error"
        :message message
        :details details
        :duration-ms duration})
      (status http-status)))

(defn wrap-error-handler
  "Centralized exception handling middleware."
  [handler]

  (fn [req]
    (try
      (handler req)
      (catch AssertionError err
        (let [duration (- (System/currentTimeMillis)
                          (:start-time req))]

          (log/warn
           {:msg "Validation failed"
            :error (.getMessage err)
            :metric {:duration-ms duration}})
          (error-response
           400
           "Validation failed"
           duration
           (.getMessage err))))
      (catch clojure.lang.ExceptionInfo err
        (let [duration (- (System/currentTimeMillis) (:start-time req))
              err-data (ex-data err)
              error-type (:type err-data)
              message
              (case error-type

                :dataset/read-failed
                "Unable to read dataset"

                :vector/vector-creation-failed
                "Feature vector creation failed"

                :spark/session-create-failed
                "Spark session creation failed"

                :spark/warmup-failed
                "Spark warmup failed"

                "Application error")]

          (log/error
           {:msg message
            :type error-type
            :details err-data
            :error (.getMessage err)
            :metric {:duration-ms duration}})

          (error-response
           500
           message
           duration
           err-data)))

      (catch Exception err
        (let [duration (- (System/currentTimeMillis) (:start-time req))]
          (log/error {:msg "Unhandled exception"
                      :error (.getMessage err)
                      :exception err
                      :metric {:duration-ms duration}})
          (error-response
           500
           "Internal server error"
           duration
           (.getMessage err)))))))

(defn wrap-request-timer
  "Attaches start-time to request for duration tracking."
  [handler]
  (fn [req]
    (handler (assoc req :start-time (System/currentTimeMillis)))))