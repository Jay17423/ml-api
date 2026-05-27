(ns ml-api.services.dataset
  (:require
   [taoensso.timbre :as log]))

(defn read-dataset
  "Loads dataset into Spark from CSV."
  [spark {:keys [url options]}]
  (let [opts (merge {:header true :delimiter ","} options)]
    (try
      (log/info {:msg "Reading dataset"
                 :metric {:path url}})
      (-> spark
          .read
          (.format "csv")
          (.option "header" (if (:header opts)
                              "true" "false"))
          (.option "delimiter" (if (:delimiter opts)
                                 (:delimiter opts) ","))
          (.load url))
      
      (catch Exception err
        (throw
         (ex-info "Failed to read dataset" {:type :dataset/read-failed
                                            :path url
                                            :error (.getMessage err)} err))))))