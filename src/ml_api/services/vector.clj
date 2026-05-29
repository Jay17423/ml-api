(ns ml-api.services.vector
  (:require
   [taoensso.timbre :as log])
  (:import
   [org.apache.spark.ml.feature VectorAssembler]))

(defn create-feature-vector
  "Creates feature vector column using Spark VectorAssembler."
  [dataset feature-cols]
  (try
    (log/info
     {:msg "Creating feature vectors"
      :metric {:feature-cols feature-cols}})
    (let [assembler (-> (VectorAssembler.)
                        (.setInputCols (into-array String feature-cols))
                        (.setOutputCol "features"))]
      (log/info
       {:msg "Feature vectors created successfully"})
      (.transform assembler dataset))
    (catch Exception err
      (throw (ex-info "Feature vector creation failed"
                      {:type :vector/vector-creation-failed
                       :feature-cols feature-cols
                       :error (.getMessage err)}
                      err)))))