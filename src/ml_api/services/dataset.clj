(ns ml-api.services.dataset
  (:require
   [taoensso.timbre :as log]))

(defn read-dataset
  "Reads CSV dataset generically for all algorithms."
  [spark {:keys [url options]}]
  (let [opts (merge {:header true
                     :delimiter ","
                     :infer-schema true}
                    options)]

    (try
      (log/info {:msg "Reading dataset"
                 :url url
                 :options opts})
      (-> spark
          .read
          (.format "csv")
          (.option "header" (str (:header opts)))
          (.option "delimiter" (str (:delimiter opts)))
          (.option "inferSchema" (str (:infer-schema opts)))
          (.load url))
      (catch Exception err
        (throw (ex-info "Dataset read failed"
                        {:type :dataset/read-failed
                         :url url
                         :error (.getMessage err)}
                        err))))))