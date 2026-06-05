(ns ml-api.core
  "Main entry point for ML API request handling and algorithm execution."
  (:gen-class)
  (:require
   [mount.core :as mount]
   [compojure.core :refer [defroutes POST]]
   [compojure.route :as route]
   [ring.util.response :refer [response status]]
   [ring.adapter.jetty :as jetty]
   [ring.middleware.json :refer [wrap-json-body wrap-json-response]]
   [ring.middleware.reload :refer [wrap-reload]]
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
   [ml-api.algorithms.binarizer :as bin]
   [ml-api.algorithms.normalizer :as norm]
   [ml-api.algorithms.standard-scaler :as ss]
   [ml-api.algorithms.bucketizer :as bucket]
   [ml-api.algorithms.imputer :as imp]
   [ml-api.algorithms.chi-sq-selector :as css]
   [ml-api.algorithms.bucketed-random-projection-lsh :as brp]
   [ml-api.algorithms.glr :as glr]
   [ml-api.algorithms.lda :as lda]
   [ml-api.algorithms.glr-model :as glr-model]
   [ml-api.preprocessing :as preprocess]
   [taoensso.timbre :as log]
   [omniconf.core :as cfg]
   [ml-api.specs :as specs]))

"TODO : Let for let variable in all the algorithm and remove where it is not 
 needed."

(defn execute-algorithm
  "Dispatches ML algorithm execution."
  [algo ds params]

  (case algo
    "ChiSquareTest"
    (chi/execute ds params)

    "CountVectorizer"
    (cv/execute ds params)

    "Tokenizer"
    (tokenizer/execute-tokenizer ds params)

    "RegexTokenizer"
    (tokenizer/execute-regex-tokenizer ds params)

    "StopWordsRemover"
    (swr/execute ds params)

    "NGram"
    (ngram/execute ds params)

    "StringIndexer"
    (si/execute ds params)

    "Binarizer"
    (bin/execute ds params)

    "Normalizer"
    (norm/execute ds params)

    "StandardScaler"
    (ss/execute ds params)

    "Bucketizer"
    (bucket/execute ds params)

    "Imputer"
    (imp/execute ds params)

    "ChiSqSelector"
    (css/execute ds params)

    "BucketedRandomProjectionLSH"
    (brp/execute ds params)

    "GeneralizedLinearRegression"
    (glr/execute ds params)

    "LDA"
    (lda/execute ds params)

    "GeneralizedLinearRegressionModel"
    (glr-model/execute ds params)

    (throw (ex-info "Unsupported algorithm"
                    {:type :algorithm/not-supported
                     :algorithm algo}))))

(defn execute
  "Handles ML execution request."
  [req]
  (specs/validate-request (:body req))
  (let [{:keys [algorithm parameters]} (:body req)
        _ (log/info {:msg "ML request received"
                     :algorithm algorithm})
        raw-ds (ds/read-dataset session parameters)
        processed-ds (preprocess/preprocess-dataset raw-ds parameters)
        result (execute-algorithm algorithm processed-ds parameters)
        duration (- (System/currentTimeMillis) (:start-time req))]
    (log/info {:msg "ML execution completed"
               :algorithm algorithm
               :metric {:duration-ms duration}})
    (-> (response {:status "success"
                   :duration-ms duration
                   :result result
                   :parameters parameters})
        (status 200))))

(defroutes app-routes
  "Define API routes."
  (POST "/v1/ml/execute" [] execute)
  (route/not-found (status (response {:error "NOT_FOUND"})
                           404)))

(def app
  "Ring application with middleware."
  (-> app-routes
      (mw/wrap-error-handler)
      (mw/wrap-request-timer)
      (wrap-json-body {:keywords? true})
      wrap-json-response
      wrap-reload))

(defn -main
  "Starts application and HTTP server."
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