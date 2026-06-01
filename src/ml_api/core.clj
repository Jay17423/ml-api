(ns ml-api.core
  (:gen-class)
  (:require
   [mount.core :as mount]
   [compojure.core :refer [defroutes POST]]
   [compojure.route :as route]
   [ring.util.response :refer [response status]]
   [ring.adapter.jetty :as jetty]
   [ring.middleware.json :refer [wrap-json-body wrap-json-response]]
   [ml-api.logger :as logger]
   [ml-api.middleware :as mw]
   [ml-api.state :refer [session]]
   [ml-api.services.dataset :as ds]
   [ml-api.algorithms.chi-square-test :as chi]
   [ml-api.algorithms.count-vectorizer :as cv]
   [ml-api.algorithms.tokenizer :as tokenizer]
   [ml-api.algorithms.stop-words-remover :as swr]
   [ml-api.algorithms.ngram :as ngram]
   [ml-api.algorithms.string-indexer :as si]
   [taoensso.timbre :as log]
   [omniconf.core :as cfg]
   [ml-api.specs :as specs]))

"TODO : change all underscore params name to kebab case after completing the 
 project"

(defn execute-algorithm
  "Dispatches ML algorithm execution."
  [algorithm dataset parameters]

  (case algorithm
    "ChiSquareTest"
    (chi/execute dataset parameters)

    "CountVectorizer"
    (cv/execute dataset parameters)

    "Tokenizer"
    (tokenizer/execute-tokenizer
     dataset
     parameters)
    
    "RegexTokenizer"
    (tokenizer/execute-regex-tokenizer
     dataset
     parameters)
    
    "StopWordsRemover"
    (swr/execute dataset parameters)

    "NGram"
    (ngram/execute dataset parameters)

    "StringIndexer"
    (si/execute dataset parameters)

    (throw (ex-info "Unsupported algorithm"
                    {:type :algorithm/not-supported
                     :algorithm algorithm}))))

(defn execute
  "Handles ML execution request."
  [req]
   (specs/validate-request (:body req))
  (let [{:keys [algorithm parameters]} (:body req) 
        _ (log/info {:msg "ML request received"
                     :algorithm algorithm})
        dataset (ds/read-dataset session parameters)
        result (execute-algorithm algorithm dataset parameters)
        duration (- (System/currentTimeMillis) (:start-time req))]
    (log/info
     {:msg "ML execution completed"
      :algorithm algorithm
      :metric {:duration-ms duration}})

    (-> (response
         {:status "success"
          :duration-ms duration
          :result (:data result)
          :parameters parameters })
        (status 200))))

(defroutes app-routes
  (POST "/v1/ml/execute" [] execute)
  (route/not-found
   (status (response {:error "NOT_FOUND"})
           404)))

(def app
  (-> app-routes
      (mw/wrap-error-handler)
      (mw/wrap-request-timer)
      (wrap-json-body {:keywords? true})
      wrap-json-response))

(defn -main
  []
  (try
    (cfg/populate-from-file
     "config/config.edn")
    (cfg/verify)
    (logger/configure-logger!)
    (log/info
     {:msg "Starting ml-api service"})
    (mount/start)
    (log/info
     {:msg "Starting http server"
      :metric {:port (cfg/get :server :port)}})
    (jetty/run-jetty app {:port (cfg/get :server :port)})
    (catch Exception err
      (log/error {:msg "Startup failed"
                  :error (.getMessage err)})
      (System/exit 1))))