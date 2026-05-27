(ns ml-api.core
  (:gen-class)
  (:require [mount.core :as mount]
            [compojure.core :refer [defroutes POST]]
            [compojure.route :as route]
            [ring.util.response :refer [response status]]
            [ring.adapter.jetty :as jetty]
            [ring.middleware.json :refer [wrap-json-body wrap-json-response]]
            [ml-api.logger :as logger]
            [taoensso.timbre :as log]
            [omniconf.core :as cfg]
            [ml-api.middleware :as mw]
            [ml-api.state :refer [session]]
            [ml-api.utils :as util]
            [ml-api.services.vector :as vs]
            [ml-api.services.dataset :as ds]))

(defn execute
  "Handles dataset load request."

  [req]

  (log/info
   {:msg "Dataset load request"
    :metric {:type (:type (:body req))}})

  (let [body (:body req)
        dataset (ds/read-dataset session body)
        vectorized-dataset (vs/create-feature-vector dataset (:feature_field body))
        duration (- (System/currentTimeMillis) (:start-time req))
        preview (util/dataset->preview vectorized-dataset)]

    (log/info
     {:msg "Dataset processed successfully"
      :metric {:duration-ms duration}})

    (status
     (response {:status "success"
                :duration-ms duration
                :data preview}) 
     200)))

(defroutes app-routes
  "Defines API routes."
  (POST "/v1/ml/execute" [] execute )
  (route/not-found (status (response {:error "NOT_FOUND"}) 404)))

(def app
  "Ring application with JSON middleware."
  (-> app-routes 
      (mw/wrap-error-handler)
      (mw/wrap-request-timer)
      (wrap-json-body {:keywords? true})
      wrap-json-response))

(defn -main
  "Starts application and HTTP server."
  []
  (try
    (cfg/populate-from-file "config/config.edn")
    (cfg/verify)
    (logger/configure-logger!)
    (log/info {:msg "Starting connector service"})
    (mount/start)
    (log/info {:msg "Starting http server"
               :metric {:port (cfg/get :server :port)}})
    (jetty/run-jetty app {:port (cfg/get :server :port)})
    (catch Exception err
      (log/error {:msg "Startup failed"
                  :error (.getMessage err)})
      (System/exit 1))))